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

            # 5. Generate questions for each concept (in parallel batches)
            total_questions = 0
            failed_concepts = []

            logger.info(f"Starting parallel question generation for {len(concepts)} concepts (max {MAX_CONCURRENT_GENERATIONS} concurrent)")

            # Process concepts in parallel batches
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)

            async def process_concept(concept: Dict[str, Any], concept_index: int) -> Tuple[List[GeneratedQuestion], Optional[str]]:
                """Process a single concept with semaphore-controlled concurrency."""
                async with semaphore:
                    logger.info(f"Processing concept {concept_index}/{len(concepts)}: {concept['name']}")
                    try:
                        # Determine number of questions dynamically
                        num_questions = self.question_generator.calculate_questions_per_concept(
                            concept_explanation=concept.get("explanation", ""),
                            source_text=concept.get("source_text", ""),
                            min_questions=min_questions_per_concept,
                            max_questions=max_questions_per_concept
                        )

                        # Generate questions
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
            for (questions, failed_concept_name) in results:
                if failed_concept_name:
                    failed_concepts.append(failed_concept_name)
                    logger.warning(f"No questions generated for concept: {failed_concept_name}")
                elif questions:
                    saved_count = await self._save_questions(quiz_id, questions, current_order_index)
                    current_order_index += saved_count
                    total_questions += saved_count
                    logger.info(f"Saved {saved_count} questions")

            # 6. Update quiz status (ASYNC)
            if total_questions > 0:
                await self._update_quiz_status(quiz_id, "completed")
                logger.info(
                    f"Quiz generation completed: {total_questions} questions generated "
                    f"({len(failed_concepts)} concepts failed)"
                )
            else:
                await self._update_quiz_status(quiz_id, "failed")
                logger.error("Quiz generation failed: no questions generated")
                return None

            return quiz_id

        except Exception as e:
            logger.error(f"Quiz generation failed for document {document_id}: {e}")
            try:
                if quiz_id:
                    await self._update_quiz_status(quiz_id, "failed")
            except:
                pass
            return None

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

    async def _update_quiz_status(self, quiz_id: str, status: str) -> bool:
        """Update quiz generation status (ASYNC)."""
        try:
            await run_db_operation(
                lambda: self.supabase.table("quizzes").update({
                    "generation_status": status
                }).eq("id", quiz_id).execute()
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
