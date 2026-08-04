import logging
import asyncio
import random
from typing import List, Dict, Any, Optional, Tuple
from app.core.supabase import get_supabase
from app.core.async_db import run_db_operation
from app.services.question_generator import QuestionGenerator
from app.schemas.quiz import GeneratedQuestion

logger = logging.getLogger(__name__)

# Concurrency limit for parallel question generation
MAX_CONCURRENT_GENERATIONS = 3

# Hard ceiling on total questions generated per document. This bank is a
# byproduct of the auto-quiz-on-upload step (never served to students directly
# — see extraction_service.py) so it exists purely to signal "materials are
# ready"; uncapped, a content-rich document can produce 100+ questions of
# pure Sonnet spend for no product benefit.
MAX_TOTAL_QUESTIONS_PER_DOCUMENT = 30

# Overall time budget for one document's generation run. Wrapped around the
# generation work itself (not by the caller) so a timeout is caught by this
# function's own except block below and reliably marks the quiz 'failed'.
QUIZ_GENERATION_TIMEOUT = 900


class QuizGenerationService:
    """
    Orchestrates quiz generation from document concepts.
    Coordinates between concept extraction and question generation.
    All database operations are async to prevent blocking.
    """

    def __init__(self):
        self.supabase = get_supabase()
        self.question_generator = QuestionGenerator()

    async def generate_quiz_for_document(
        self,
        document_id: str,
        user_id: str,
        min_questions_per_concept: int = 2,
        max_questions_per_concept: int = 5
    ) -> Optional[str]:
        """
        Generate a complete quiz from a document's extracted concepts.

        Args:
            document_id: UUID of the processed document
            user_id: UUID of the user who owns the document
            min_questions_per_concept: Minimum questions per concept
            max_questions_per_concept: Maximum questions per concept

        Returns:
            quiz_id if successful, None if failed
        """
        quiz_id = None
        try:
            logger.info(f"Starting quiz generation for document {document_id}")

            # 1. Verify document exists and is completed (ASYNC)
            doc = await self._get_document(document_id)
            if not doc:
                logger.error(f"Document {document_id} not found")
                return None

            if doc.get("status") != "completed":
                logger.error(f"Document {document_id} not yet processed")
                return None

            # 2. Get all concepts for this document (ASYNC)
            concepts = await self._get_document_concepts(document_id)
            if not concepts:
                logger.warning(f"No concepts found for document {document_id}")
                return None

            logger.info(f"Found {len(concepts)} concepts for quiz generation")

            # 3. Create quiz record (ASYNC)
            quiz_id = await self._create_quiz(
                document_id=document_id,
                user_id=user_id,
                document_title=doc.get("title", "Untitled"),
                concept_count=len(concepts)
            )

            if not quiz_id:
                logger.error("Failed to create quiz record")
                return None

            # 4. Update status to generating (ASYNC)
            await self._update_quiz_status(quiz_id, "generating")

            # 5. Generate questions for each concept, bounded by both a total
            # question budget and an overall time budget. Wrapped in wait_for
            # *here* (not by the caller) so a timeout raises inside this try
            # block and is caught below, which reliably marks the quiz
            # 'failed' — a timeout enforced by the caller instead would cancel
            # this coroutine via CancelledError, which bypasses this handling.
            total_questions = await asyncio.wait_for(
                self._run_generation(
                    quiz_id, concepts, min_questions_per_concept, max_questions_per_concept
                ),
                timeout=QUIZ_GENERATION_TIMEOUT,
            )

            # 6. Update quiz status (ASYNC)
            if total_questions > 0:
                await self._update_quiz_status(quiz_id, "completed")
                logger.info(f"Quiz generation completed: {total_questions} questions generated")
            else:
                await self._update_quiz_status(
                    quiz_id, "failed", error_message="No questions could be generated for this document."
                )
                logger.error("Quiz generation failed: no questions generated")
                return None

            return quiz_id

        except asyncio.TimeoutError:
            logger.error(f"Quiz generation timed out after {QUIZ_GENERATION_TIMEOUT}s for document {document_id}")
            try:
                if quiz_id:
                    await self._update_quiz_status(
                        quiz_id, "failed", error_message=f"Generation timed out after {QUIZ_GENERATION_TIMEOUT}s."
                    )
            except Exception:
                pass
            return None

        except Exception as e:
            logger.error(f"Quiz generation failed for document {document_id}: {e}")
            try:
                if quiz_id:
                    await self._update_quiz_status(quiz_id, "failed", error_message=str(e)[:500])
            except Exception:
                pass
            return None

    async def _run_generation(
        self,
        quiz_id: str,
        concepts: List[Dict[str, Any]],
        min_questions_per_concept: int,
        max_questions_per_concept: int,
    ) -> int:
        """
        Generate and save questions for all concepts, in parallel batches,
        capped at MAX_TOTAL_QUESTIONS_PER_DOCUMENT total questions.

        Returns the total number of questions saved.
        """
        logger.info(
            f"Starting parallel question generation for {len(concepts)} concepts "
            f"(max {MAX_CONCURRENT_GENERATIONS} concurrent, budget {MAX_TOTAL_QUESTIONS_PER_DOCUMENT} questions)"
        )

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
        budget_lock = asyncio.Lock()
        questions_remaining = MAX_TOTAL_QUESTIONS_PER_DOCUMENT

        async def process_concept(concept: Dict[str, Any], concept_index: int) -> Tuple[List[GeneratedQuestion], Optional[str]]:
            """Process a single concept with semaphore-controlled concurrency."""
            nonlocal questions_remaining
            async with semaphore:
                # Reserve this concept's share of the total question budget
                # up front so the cap holds regardless of concept order or
                # how many concepts run concurrently.
                async with budget_lock:
                    if questions_remaining <= 0:
                        logger.info(f"Question budget exhausted, skipping concept: {concept['name']}")
                        return ([], None)

                    num_questions = min(
                        self.question_generator.calculate_questions_per_concept(
                            concept_explanation=concept.get("explanation", ""),
                            source_text=concept.get("source_text", ""),
                            min_questions=min_questions_per_concept,
                            max_questions=max_questions_per_concept
                        ),
                        questions_remaining,
                    )
                    questions_remaining -= num_questions

                logger.info(f"Processing concept {concept_index}/{len(concepts)}: {concept['name']}")
                try:
                    questions = await self.question_generator.generate_questions_for_concept(
                        concept_id=concept["id"],
                        concept_name=concept["name"],
                        concept_explanation=concept.get("explanation", ""),
                        source_text=concept.get("source_text", ""),
                        num_questions=num_questions
                    )

                    if questions:
                        return (questions, None)
                    else:
                        return ([], concept["name"])

                except Exception as e:
                    logger.error(f"Failed to generate questions for concept {concept['name']}: {e}")
                    return ([], concept["name"])

        # Run all concepts in parallel with controlled concurrency
        tasks = [
            process_concept(concept, i + 1)
            for i, concept in enumerate(concepts)
        ]
        results = await asyncio.gather(*tasks)

        # Process results and save questions (ASYNC)
        current_order_index = 0
        total_questions = 0
        failed_concepts = []
        for (questions, failed_concept_name) in results:
            if failed_concept_name:
                failed_concepts.append(failed_concept_name)
                logger.warning(f"No questions generated for concept: {failed_concept_name}")
            elif questions:
                saved_count = await self._save_questions(quiz_id, questions, current_order_index)
                current_order_index += saved_count
                total_questions += saved_count
                logger.info(f"Saved {saved_count} questions")

        if failed_concepts:
            logger.info(f"{len(failed_concepts)} concept(s) failed to generate questions")

        return total_questions

    async def _get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Fetch document from database (ASYNC)."""
        try:
            response = await run_db_operation(
                lambda: self.supabase.table("documents").select("*").eq("id", document_id).single().execute()
            )
            return response.data if response.data else None
        except Exception as e:
            logger.error(f"Error fetching document {document_id}: {e}")
            return None

    async def _get_document_concepts(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Fetch all concepts for a document (via topics) (ASYNC).
        Returns list of concept dictionaries.
        """
        try:
            # Get all topics for this document
            topics_response = await run_db_operation(
                lambda: self.supabase.table("topics").select("id").eq("document_id", document_id).execute()
            )

            if not topics_response.data:
                return []

            topic_ids = [topic["id"] for topic in topics_response.data]

            # Get all concepts for these topics
            concepts_response = await run_db_operation(
                lambda: self.supabase.table("concepts").select("*").in_("topic_id", topic_ids).execute()
            )

            return concepts_response.data if concepts_response.data else []

        except Exception as e:
            logger.error(f"Error fetching concepts for document {document_id}: {e}")
            return []

    async def _create_quiz(
        self,
        document_id: str,
        user_id: str,
        document_title: str,
        concept_count: int
    ) -> Optional[str]:
        """Create a new quiz record in the database (ASYNC)."""
        try:
            quiz_title = f"Quiz: {document_title}"
            quiz_description = (
                f"Automatically generated quiz covering {concept_count} concepts "
                f"from your uploaded material."
            )

            response = await run_db_operation(
                lambda: self.supabase.table("quizzes").insert({
                    "title": quiz_title,
                    "description": quiz_description,
                    "document_id": document_id,
                    "user_id": user_id,
                    "generation_status": "pending"
                }).execute()
            )

            if response.data and isinstance(response.data, list) and len(response.data) > 0:
                return response.data[0]["id"]

            return None

        except Exception as e:
            logger.error(f"Error creating quiz: {e}")
            return None

    async def _update_quiz_status(
        self, quiz_id: str, status: str, error_message: Optional[str] = None
    ) -> bool:
        """Update quiz generation status, optionally recording why it failed (ASYNC)."""
        try:
            update: Dict[str, Any] = {"generation_status": status}
            if error_message is not None:
                update["error_message"] = error_message
            await run_db_operation(
                lambda: self.supabase.table("quizzes").update(update).eq("id", quiz_id).execute()
            )
            return True
        except Exception as e:
            logger.error(f"Error updating quiz status: {e}")
            return False

    async def _save_questions(
        self,
        quiz_id: str,
        questions: List[GeneratedQuestion],
        start_order_index: int
    ) -> int:
        """
        Save generated questions and their options to database (ASYNC).
        Shuffles option order to randomize correct answer position.
        Returns count of successfully saved questions.
        """
        saved_count = 0

        for i, question in enumerate(questions):
            try:
                # Create a list of options with their original data
                options_list = list(question.options)

                # Shuffle the options to randomize correct answer position
                random.shuffle(options_list)

                # Find the new index of the correct answer after shuffling
                correct_answer_index = next(
                    new_idx for new_idx, opt in enumerate(options_list) if opt.is_correct
                )

                # Insert question record with the new correct answer index (ASYNC)
                question_response = await run_db_operation(
                    lambda q=question, idx=i, ca_idx=correct_answer_index: self.supabase.table("questions").insert({
                        "quiz_id": quiz_id,
                        "question": q.question,
                        "options": [],  # Legacy field, keep empty
                        "correct_answer": ca_idx,
                        "explanation": "",  # Legacy field, keep empty
                        "order_index": start_order_index + idx,
                        "concept_id": q.concept_id,
                        "hint": q.hint,
                        "difficulty_level": q.difficulty_level
                    }).execute()
                )

                if not question_response.data or len(question_response.data) == 0:
                    logger.error(f"Failed to insert question: {question.question[:50]}")
                    continue

                question_id = question_response.data[0]["id"]

                # Insert question options with new shuffled indices (ASYNC)
                options_data = [
                    {
                        "question_id": question_id,
                        "option_text": opt.option_text,
                        "option_index": new_idx,  # Use new shuffled index
                        "is_correct": opt.is_correct,
                        "explanation": opt.explanation
                    }
                    for new_idx, opt in enumerate(options_list)
                ]

                await run_db_operation(
                    lambda od=options_data: self.supabase.table("question_options").insert(od).execute()
                )
                saved_count += 1

            except Exception as e:
                logger.error(f"Error saving question: {e}")
                continue

        return saved_count
