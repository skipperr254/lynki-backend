"""
Quiz sessions endpoint.

POST /quiz-sessions/generate
  Generates a fresh named quiz (course_quizzes record) with BKT-guided questions.
  Returns {quiz_id, name, total_questions, course_id} — not TestData.
  The frontend then navigates to TestPage which starts an attempt.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services.on_demand_quiz_service import generate_quiz
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class GenerateQuizRequest(BaseModel):
    user_id: str
    course_id: str
    quiz_size: int = Field(default=10, ge=1, le=30)
    document_id: Optional[str] = None


@router.post("/generate")
async def generate_quiz_endpoint(req: GenerateQuizRequest):
    try:
        result = await generate_quiz(
            user_id=req.user_id,
            course_id=req.course_id,
            quiz_size=req.quiz_size,
            document_id=req.document_id,
        )
        if result.get("error"):
            raise HTTPException(status_code=422, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quiz generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate quiz")
