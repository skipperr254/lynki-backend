"""
Quiz sessions endpoint.

POST /quiz-sessions/generate
  Kicks off generation of a fresh named quiz (course_quizzes record) with
  BKT-guided questions and returns immediately — {quiz_id, status, course_id},
  status='generating'. Actual question generation runs as a background task
  (bounded by QUIZ_GENERATION_TIMEOUT in on_demand_quiz_service) so this
  request never blocks on Sonnet calls. The frontend polls
  course_quizzes.status directly via Supabase until it reaches a terminal
  state ('completed' or 'failed'), then starts an attempt via
  quiz_attempts_service.
"""

from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from app.services.on_demand_quiz_service import (
    start_quiz_generation,
    run_quiz_generation_job,
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class GenerateQuizRequest(BaseModel):
    user_id: str
    course_id: str
    quiz_size: int = Field(default=10, ge=1, le=30)
    document_id: Optional[str] = None


@router.post("/generate")
async def generate_quiz_endpoint(req: GenerateQuizRequest, background_tasks: BackgroundTasks):
    try:
        result = await start_quiz_generation(
            user_id=req.user_id,
            course_id=req.course_id,
            quiz_size=req.quiz_size,
            document_id=req.document_id,
        )
        if result.get("error"):
            raise HTTPException(status_code=422, detail=result["error"])

        concepts = result.pop("concepts")
        background_tasks.add_task(
            run_quiz_generation_job,
            quiz_id=result["quiz_id"],
            user_id=req.user_id,
            course_id=req.course_id,
            concepts=concepts,
            quiz_size=req.quiz_size,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quiz generation failed to start: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate quiz")
