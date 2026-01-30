from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from app.schemas.quiz import (
    QuizListItem,
    QuizResponse,
    QuestionResponse,
    QuestionOptionResponse,
    QuizAttemptSubmit,
    QuizAttemptResult,
    QuestionResult,
    QuizGenerationRequest,
    QuizGenerationResponse
)
from app.core.supabase import get_supabase
from app.services.quiz_generation_service import QuizGenerationService
import logging

router = APIRouter()
supabase = get_supabase()
quiz_service = QuizGenerationService()


@router.get("/user/{user_id}", response_model=List[QuizListItem])
async def list_user_quizzes(user_id: str):
    """
    Get all quizzes for a user.
    Returns list with summary info (no questions).
    """
    try:
        # Use the quiz_summary view for efficient querying
        response = supabase.table("quiz_summary").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()  # type: ignore
        
        if not response.data:
            return []
        
        quizzes = []
        for quiz in response.data:  # type: ignore[attr-defined]
            quizzes.append(QuizListItem(
                id=quiz["id"],  # type: ignore[index]
                title=quiz["title"],  # type: ignore[index]
                description=quiz["description"],  # type: ignore[index]
                document_id=quiz.get("document_id"),  # type: ignore[attr-defined]
                document_title=quiz.get("document_title"),  # type: ignore[attr-defined]
                generation_status=quiz["generation_status"],  # type: ignore[index]
                question_count=quiz.get("question_count", 0),  # type: ignore[attr-defined]
                created_at=quiz["created_at"]  # type: ignore[index]
            ))
        
        return quizzes
        
    except Exception as e:
        logging.error(f"Error fetching user quizzes: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch quizzes")


@router.get("/document/{document_id}", response_model=Optional[QuizListItem])
async def get_document_quiz(document_id: str):
    """
    Get the quiz for a specific document.
    Returns None if no quiz exists yet.
    """
    try:
        response = supabase.table("quiz_summary").select("*").eq("document_id", document_id).execute()  # type: ignore
        
        if not response.data or len(response.data) == 0:  # type: ignore[arg-type]
            return None
        
        quiz = response.data[0]  # type: ignore[index]
        return QuizListItem(
            id=quiz["id"],  # type: ignore[index]
            title=quiz["title"],  # type: ignore[index]
            description=quiz["description"],  # type: ignore[index]
            document_id=quiz.get("document_id"),  # type: ignore[attr-defined]
            document_title=quiz.get("document_title"),  # type: ignore[attr-defined]
            generation_status=quiz["generation_status"],  # type: ignore[index]
            question_count=quiz.get("question_count", 0),  # type: ignore[attr-defined]
            created_at=quiz["created_at"]  # type: ignore[index]
        )
        
    except Exception as e:
        logging.error(f"Error fetching document quiz: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch quiz")


@router.get("/{quiz_id}", response_model=QuizResponse)
async def get_quiz(quiz_id: str):
    """
    Get full quiz details including all questions and options.
    """
    try:
        # Fetch quiz
        quiz_response = supabase.table("quizzes").select("*").eq("id", quiz_id).single().execute()  # type: ignore
        
        if not quiz_response.data:
            raise HTTPException(status_code=404, detail="Quiz not found")
        
        quiz_data = quiz_response.data
        
        # Fetch questions
        questions_response = supabase.table("questions").select("*").eq("quiz_id", quiz_id).order("order_index").execute()  # type: ignore
        
        questions = []
        if questions_response.data:
            for q in questions_response.data:  # type: ignore[attr-defined]
                question_id = q["id"]  # type: ignore[index]
                
                # Fetch options for this question
                options_response = supabase.table("question_options").select("*").eq("question_id", question_id).order("option_index").execute()  # type: ignore
                
                options = []
                if options_response.data:
                    for opt in options_response.data:  # type: ignore[attr-defined]
                        options.append(QuestionOptionResponse(
                            id=opt["id"],  # type: ignore[index]
                            option_text=opt["option_text"],  # type: ignore[index]
                            option_index=opt["option_index"],  # type: ignore[index]
                            is_correct=opt["is_correct"],  # type: ignore[index]
                            explanation=opt["explanation"]  # type: ignore[index]
                        ))
                
                questions.append(QuestionResponse(
                    id=question_id,  # type: ignore[arg-type]
                    question=q["question"],  # type: ignore[index]
                    options=options,
                    hint=q.get("hint"),  # type: ignore[attr-defined]
                    difficulty_level=q["difficulty_level"],  # type: ignore[index]
                    concept_id=q.get("concept_id"),  # type: ignore[attr-defined]
                    order_index=q["order_index"]  # type: ignore[index]
                ))
        
        return QuizResponse(
            id=quiz_data["id"],  # type: ignore[index]
            title=quiz_data["title"],  # type: ignore[index]
            description=quiz_data["description"],  # type: ignore[index]
            document_id=quiz_data.get("document_id"),  # type: ignore[attr-defined]
            user_id=quiz_data.get("user_id"),  # type: ignore[attr-defined]
            generation_status=quiz_data["generation_status"],  # type: ignore[index]
            questions=questions,
            created_at=quiz_data["created_at"],  # type: ignore[index]
            updated_at=quiz_data["updated_at"]  # type: ignore[index]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching quiz: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch quiz")


@router.post("/attempt/submit", response_model=QuizAttemptResult)
async def submit_quiz_attempt(attempt: QuizAttemptSubmit, user_id: str):
    """
    Submit a quiz attempt and get results.
    Saves attempt to database and returns detailed results.
    """
    try:
        # Fetch quiz with questions
        quiz_response = supabase.table("quizzes").select("*").eq("id", attempt.quiz_id).single().execute()  # type: ignore
        
        if not quiz_response.data:
            raise HTTPException(status_code=404, detail="Quiz not found")
        
        # Fetch all questions with options
        questions_response = supabase.table("questions").select("*").eq("quiz_id", attempt.quiz_id).execute()  # type: ignore
        
        if not questions_response.data:
            raise HTTPException(status_code=404, detail="No questions found for quiz")
        
        # Build question lookup with options
        question_map = {}
        for q in questions_response.data:  # type: ignore[attr-defined]
            question_id = q["id"]  # type: ignore[index]
            
            # Fetch options
            options_response = supabase.table("question_options").select("*").eq("question_id", question_id).order("option_index").execute()  # type: ignore
            
            question_map[question_id] = {
                "question": q,  # type: ignore[index]
                "options": options_response.data if options_response.data else []
            }
        
        # Grade the attempt
        score = 0
        total_questions = len(attempt.answers)
        question_results = []
        
        for answer in attempt.answers:
            if answer.question_id not in question_map:
                continue
            
            q_data = question_map[answer.question_id]
            question = q_data["question"]
            options = q_data["options"]
            
            # Find correct answer
            correct_option = next(
                (opt for opt in options if opt["is_correct"]),  # type: ignore[index]
                None
            )
            correct_index = correct_option["option_index"] if correct_option else -1  # type: ignore[index]
            
            is_correct = answer.selected_option_index == correct_index
            if is_correct:
                score += 1
            
            # Get explanation for selected option
            selected_option = next(
                (opt for opt in options if opt["option_index"] == answer.selected_option_index),  # type: ignore[index]
                None
            )
            explanation = selected_option["explanation"] if selected_option else "No explanation available"  # type: ignore[index]
            
            question_results.append(QuestionResult(
                question_id=answer.question_id,
                question_text=question["question"],  # type: ignore[index]
                selected_option_index=answer.selected_option_index,
                correct_option_index=correct_index,
                is_correct=is_correct,
                explanation=explanation,  # type: ignore[arg-type]
                hint=question.get("hint")  # type: ignore[attr-defined]
            ))
        
        percentage = (score / total_questions * 100) if total_questions > 0 else 0
        
        # Save attempt to database
        attempt_data = {
            "user_id": user_id,
            "quiz_id": attempt.quiz_id,
            "score": score,
            "total_questions": total_questions,
            "answers": [{"question_id": a.question_id, "selected_option_index": a.selected_option_index} for a in attempt.answers]
        }
        
        attempt_response = supabase.table("user_quiz_attempts").insert(attempt_data).execute()  # type: ignore
        
        attempt_id = "unknown"
        if attempt_response.data and len(attempt_response.data) > 0:  # type: ignore[arg-type]
            attempt_id = attempt_response.data[0]["id"]  # type: ignore[index]
        
        return QuizAttemptResult(
            attempt_id=attempt_id,  # type: ignore[arg-type]
            quiz_id=attempt.quiz_id,
            score=score,
            total_questions=total_questions,
            percentage=round(percentage, 1),
            question_results=question_results,
            completed_at=attempt_response.data[0]["completed_at"] if attempt_response.data else ""  # type: ignore[index]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error submitting quiz attempt: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit quiz attempt")


@router.get("/attempt/history/{user_id}/{quiz_id}")
async def get_quiz_attempts(user_id: str, quiz_id: str):
    """
    Get all attempts for a specific quiz by a user.
    Returns summary of past attempts.
    """
    try:
        response = supabase.table("user_quiz_attempts").select("*").eq("user_id", user_id).eq("quiz_id", quiz_id).order("completed_at", desc=True).execute()  # type: ignore
        
        if not response.data:
            return []
        
        attempts = []
        for attempt in response.data:  # type: ignore[attr-defined]
            total = int(attempt["total_questions"])  # type: ignore[index,arg-type]
            score = int(attempt["score"])  # type: ignore[index,arg-type]
            percentage = (score / total * 100) if total > 0 else 0
            
            attempts.append({
                "id": attempt["id"],  # type: ignore[index]
                "score": score,
                "total_questions": total,
                "percentage": round(percentage, 1),
                "completed_at": attempt["completed_at"]  # type: ignore[index]
            })
        
        return attempts
        
    except Exception as e:
        logging.error(f"Error fetching quiz attempts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch attempts")


@router.post("/generate", response_model=QuizGenerationResponse)
async def trigger_quiz_generation(
    request: QuizGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Manually trigger quiz generation for a document.
    Useful for regeneration or if automatic generation failed.
    """
    try:
        # Verify document exists
        doc_response = supabase.table("documents").select("*").eq("id", request.document_id).single().execute()  # type: ignore
        
        if not doc_response.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc = doc_response.data
        user_id = doc.get("user_id")  # type: ignore[attr-defined]
        
        if not user_id:
            raise HTTPException(status_code=400, detail="Document has no user_id")
        
        # Check if quiz already exists
        existing_quiz = supabase.table("quizzes").select("id, generation_status").eq("document_id", request.document_id).execute()  # type: ignore
        
        if existing_quiz.data and len(existing_quiz.data) > 0:  # type: ignore[arg-type]
            quiz_id = existing_quiz.data[0]["id"]  # type: ignore[index]
            status = existing_quiz.data[0]["generation_status"]  # type: ignore[index]
            
            # If already generating, don't start another
            if status == "generating":
                return QuizGenerationResponse(
                    quiz_id=quiz_id,  # type: ignore[arg-type]
                    document_id=request.document_id,
                    status="generating",  # type: ignore[arg-type]
                    total_questions=0,
                    message="Quiz generation already in progress"
                )
        
        # Trigger generation in background
        background_tasks.add_task(
            quiz_service.generate_quiz_for_document,
            document_id=request.document_id,
            user_id=user_id,  # type: ignore[arg-type]
            min_questions_per_concept=2,
            max_questions_per_concept=request.questions_per_concept
        )
        
        return QuizGenerationResponse(
            quiz_id=existing_quiz.data[0]["id"] if existing_quiz.data else "pending",  # type: ignore[index]
            document_id=request.document_id,
            status="generating",  # type: ignore[arg-type]
            total_questions=0,
            message="Quiz generation started"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error triggering quiz generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to start quiz generation")
