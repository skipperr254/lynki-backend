"""
Quiz attempts endpoints.

POST /quiz-attempts/start       — start a new attempt on a quiz
POST /quiz-attempts/answer      — submit an answer within an attempt
POST /quiz-attempts/complete    — explicitly complete an attempt
GET  /quiz-attempts/resume/{user_id}/{attempt_id} — resume in-progress attempt
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.core.auth import get_current_user_id
from app.services.quiz_attempts_service import (
    start_quiz_attempt,
    resume_quiz_attempt,
    process_quiz_answer,
    complete_quiz_attempt,
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class StartAttemptRequest(BaseModel):
    user_id: str
    quiz_id: str
    course_id: str


class AnswerRequest(BaseModel):
    user_id: str
    course_id: str
    quiz_attempt_id: str
    question_id: str
    selected_option_index: int


class CompleteAttemptRequest(BaseModel):
    user_id: str
    course_id: str
    quiz_attempt_id: str


@router.post("/start")
async def start_attempt(req: StartAttemptRequest, caller: str = Depends(get_current_user_id)):
    if req.user_id != caller:
        raise HTTPException(status_code=403, detail="Not your data")
    try:
        return await start_quiz_attempt(
            user_id=req.user_id,
            quiz_id=req.quiz_id,
            course_id=req.course_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start quiz attempt: {e}")
        raise HTTPException(status_code=500, detail="Failed to start quiz attempt")


@router.post("/answer")
async def answer_question(req: AnswerRequest, caller: str = Depends(get_current_user_id)):
    if req.user_id != caller:
        raise HTTPException(status_code=403, detail="Not your data")
    try:
        return await process_quiz_answer(
            user_id=req.user_id,
            course_id=req.course_id,
            quiz_attempt_id=req.quiz_attempt_id,
            question_id=req.question_id,
            selected_option_index=req.selected_option_index,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process quiz answer: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit answer")


@router.post("/complete")
async def complete_attempt(req: CompleteAttemptRequest, caller: str = Depends(get_current_user_id)):
    if req.user_id != caller:
        raise HTTPException(status_code=403, detail="Not your data")
    try:
        return await complete_quiz_attempt(
            user_id=req.user_id,
            course_id=req.course_id,
            quiz_attempt_id=req.quiz_attempt_id,
        )
    except Exception as e:
        logger.error(f"Failed to complete quiz attempt: {e}")
        raise HTTPException(status_code=500, detail="Failed to complete quiz attempt")


@router.get("/resume/{user_id}/{attempt_id}")
async def resume_attempt(user_id: str, attempt_id: str, caller: str = Depends(get_current_user_id)):
    if user_id != caller:
        raise HTTPException(status_code=403, detail="Not your data")
    try:
        return await resume_quiz_attempt(user_id=user_id, quiz_attempt_id=attempt_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to resume quiz attempt: {e}")
        raise HTTPException(status_code=500, detail="Failed to resume quiz attempt")
