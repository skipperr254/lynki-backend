import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from app.core.supabase import get_supabase
from app.services.question_generator import QuestionGenerator
from app.schemas.quiz import GeneratedQuestion

# Concurrency limit for parallel question generation
MAX_CONCURRENT_GENERATIONS = 3

class QuizGenerationService:
    """
    Orchestrates quiz generation from document concepts.
    Coordinates between concept extraction and question generation.
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
        try:
            logging.info(f"Starting quiz generation for document {document_id}")
            
            # 1. Verify document exists and is completed
            doc = self._get_document(document_id)
            if not doc:
                logging.error(f"Document {document_id} not found")
                return None
            
            if doc.get("status") != "completed":
                logging.error(f"Document {document_id} not yet processed")
                return None
            
            # 2. Get all concepts for this document
            concepts = self._get_document_concepts(document_id)
            if not concepts:
                logging.warning(f"No concepts found for document {document_id}")
                return None
            
            logging.info(f"Found {len(concepts)} concepts for quiz generation")
            
            # 3. Create quiz record
            quiz_id = self._create_quiz(
                document_id=document_id,
                user_id=user_id,
                document_title=doc.get("title", "Untitled"),
                concept_count=len(concepts)
            )
            
            if not quiz_id:
                logging.error("Failed to create quiz record")
                return None
            
            # 4. Update status to generating
            self._update_quiz_status(quiz_id, "generating")
            
            # 5. Generate questions for each concept (in parallel batches)
            total_questions = 0
            failed_concepts = []

            logging.info(f"Starting parallel question generation for {len(concepts)} concepts (max {MAX_CONCURRENT_GENERATIONS} concurrent)")

            # Process concepts in parallel batches
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)

            async def process_concept(concept: Dict[str, Any], concept_index: int) -> Tuple[List[GeneratedQuestion], Optional[str]]:
                """Process a single concept with semaphore-controlled concurrency."""
                async with semaphore:
                    logging.info(f"Processing concept {concept_index}/{len(concepts)}: {concept['name']}")
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
                        logging.error(f"Failed to generate questions for concept {concept['name']}: {e}")
                        return ([], concept["name"])

            # Run all concepts in parallel with controlled concurrency
            tasks = [
                process_concept(concept, i + 1)
                for i, concept in enumerate(concepts)
            ]
            results = await asyncio.gather(*tasks)

            # Process results and save questions
            current_order_index = 0
            for (questions, failed_concept_name) in results:
                if failed_concept_name:
                    failed_concepts.append(failed_concept_name)
                    logging.warning(f"No questions generated for concept: {failed_concept_name}")
                elif questions:
                    saved_count = self._save_questions(quiz_id, questions, current_order_index)
                    current_order_index += saved_count
                    total_questions += saved_count
                    logging.info(f"Saved {saved_count} questions")
            
            # 6. Update quiz status
            if total_questions > 0:
                self._update_quiz_status(quiz_id, "completed")
                logging.info(
                    f"Quiz generation completed: {total_questions} questions generated "
                    f"({len(failed_concepts)} concepts failed)"
                )
            else:
                self._update_quiz_status(quiz_id, "failed")
                logging.error("Quiz generation failed: no questions generated")
                return None
            
            return quiz_id
            
        except Exception as e:
            logging.error(f"Quiz generation failed for document {document_id}: {e}")
            try:
                if quiz_id:  # type: ignore[possibly-unbound]
                    self._update_quiz_status(quiz_id, "failed")
            except:
                pass
            return None
    
    def _get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Fetch document from database."""
        try:
            response = self.supabase.table("documents").select("*").eq("id", document_id).single().execute()  # type: ignore
            return response.data if response.data else None  # type: ignore[return-value]
        except Exception as e:
            logging.error(f"Error fetching document {document_id}: {e}")
            return None
    
    def _get_document_concepts(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Fetch all concepts for a document (via topics).
        Returns list of concept dictionaries.
        """
        try:
            # Get all topics for this document
            topics_response = self.supabase.table("topics").select("id").eq("document_id", document_id).execute()  # type: ignore
            
            if not topics_response.data:
                return []
            
            topic_ids = [topic["id"] for topic in topics_response.data]  # type: ignore[index]
            
            # Get all concepts for these topics
            concepts_response = self.supabase.table("concepts").select("*").in_("topic_id", topic_ids).execute()  # type: ignore
            
            return concepts_response.data if concepts_response.data else []  # type: ignore[return-value]
            
        except Exception as e:
            logging.error(f"Error fetching concepts for document {document_id}: {e}")
            return []
    
    def _create_quiz(
        self,
        document_id: str,
        user_id: str,
        document_title: str,
        concept_count: int
    ) -> Optional[str]:
        """Create a new quiz record in the database."""
        try:
            quiz_title = f"Quiz: {document_title}"
            quiz_description = (
                f"Automatically generated quiz covering {concept_count} concepts "
                f"from your uploaded material."
            )
            
            response = self.supabase.table("quizzes").insert({
                "title": quiz_title,
                "description": quiz_description,
                "document_id": document_id,
                "user_id": user_id,
                "generation_status": "pending"
            }).execute()  # type: ignore
            
            if response.data and isinstance(response.data, list) and len(response.data) > 0:
                return response.data[0]["id"]  # type: ignore[index,return-value]
            
            return None
            
        except Exception as e:
            logging.error(f"Error creating quiz: {e}")
            return None
    
    def _update_quiz_status(self, quiz_id: str, status: str) -> bool:
        """Update quiz generation status."""
        try:
            self.supabase.table("quizzes").update({
                "generation_status": status
            }).eq("id", quiz_id).execute()  # type: ignore
            return True
        except Exception as e:
            logging.error(f"Error updating quiz status: {e}")
            return False
    
    def _save_questions(
        self,
        quiz_id: str,
        questions: List[GeneratedQuestion],
        start_order_index: int
    ) -> int:
        """
        Save generated questions and their options to database.
        Returns count of successfully saved questions.
        """
        saved_count = 0
        
        for i, question in enumerate(questions):
            try:
                # Insert question record
                question_response = self.supabase.table("questions").insert({
                    "quiz_id": quiz_id,
                    "question": question.question,
                    "options": [],  # Legacy field, keep empty
                    "correct_answer": next(
                        opt.option_index for opt in question.options if opt.is_correct
                    ),
                    "explanation": "",  # Legacy field, keep empty
                    "order_index": start_order_index + i,
                    "concept_id": question.concept_id,
                    "hint": question.hint,
                    "difficulty_level": question.difficulty_level
                }).execute()  # type: ignore
                
                if not question_response.data or len(question_response.data) == 0:
                    logging.error(f"Failed to insert question: {question.question[:50]}")
                    continue
                
                question_id = question_response.data[0]["id"]  # type: ignore[index]
                
                # Insert question options
                options_data = [
                    {
                        "question_id": question_id,
                        "option_text": opt.option_text,
                        "option_index": opt.option_index,
                        "is_correct": opt.is_correct,
                        "explanation": opt.explanation
                    }
                    for opt in question.options
                ]
                
                self.supabase.table("question_options").insert(options_data).execute()  # type: ignore
                saved_count += 1
                
            except Exception as e:
                logging.error(f"Error saving question: {e}")
                continue
        
        return saved_count
