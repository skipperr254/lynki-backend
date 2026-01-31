from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.schemas.quiz import (
    QuizGenerationRequest,
    QuizGenerationResponse
)
from app.core.supabase import get_supabase
from app.services.quiz_generation_service import QuizGenerationService
import logging

router = APIRouter()
supabase = get_supabase()
quiz_service = QuizGenerationService()

# Note: Quiz fetching endpoints have been removed.
# The frontend fetches quiz data directly from Supabase for better performance.
# This API only handles heavy processing tasks (quiz generation).


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

        # Check document status
        if doc.get("status") != "completed":  # type: ignore[attr-defined]
            raise HTTPException(
                status_code=400,
                detail="Document must be processed before generating a quiz"
            )

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

            # If completed, allow regeneration by deleting the old quiz
            if status == "completed":
                logging.info(f"Deleting existing quiz {quiz_id} for regeneration")
                # Delete old questions and quiz
                supabase.table("questions").delete().eq("quiz_id", quiz_id).execute()  # type: ignore
                supabase.table("quizzes").delete().eq("id", quiz_id).execute()  # type: ignore

        # Trigger generation in background
        background_tasks.add_task(
            quiz_service.generate_quiz_for_document,
            document_id=request.document_id,
            user_id=user_id,  # type: ignore[arg-type]
            min_questions_per_concept=2,
            max_questions_per_concept=request.questions_per_concept
        )

        return QuizGenerationResponse(
            quiz_id="pending",
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
