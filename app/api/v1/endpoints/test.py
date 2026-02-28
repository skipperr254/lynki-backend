"""
Test endpoints — simple quiz flow: generate test, submit answers, get pass chance.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from app.services.test_service import generate_test, process_test_answer, get_pass_chance, get_test_history, complete_test_session, resume_test

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TestQuestionOption(BaseModel):
    id: str
    text: str
    index: int
    is_correct: bool
    explanation: str = ""


class TestQuestion(BaseModel):
    id: str
    question: str
    concept_id: Optional[str] = None
    concept_name: str = "Unknown"
    difficulty_level: str = "medium"
    options: List[TestQuestionOption]


class TestResponse(BaseModel):
    test_id: str
    course_id: str
    course_name: str = "Quiz"
    questions: List[TestQuestion]
    total_questions: int
    message: Optional[str] = None
    answered_count: int = 0
    correct_count: int = 0


class AnswerRequest(BaseModel):
    user_id: str
    course_id: str
    question_id: str
    selected_option_index: int
    test_id: Optional[str] = None


class AnswerResponse(BaseModel):
    question_id: str
    concept_id: Optional[str] = None
    is_correct: bool
    correct_option_index: int
    correct_option_text: str
    explanation: str = ""
    selected_option_index: int
    p_mastery_before: float
    p_mastery_after: float
    is_newly_mastered: bool = False
    mastery_threshold: float


class PassChanceSkill(BaseModel):
    skill_name: str
    mastery: float
    attempts: int


class PassChanceResponse(BaseModel):
    course_id: str
    pass_probability: float
    total_skills: int
    skills: List[PassChanceSkill]


class TestSessionResponse(BaseModel):
    id: str
    status: str
    total_questions: int
    correct_count: int
    answered_count: int
    pass_chance: Optional[float] = None
    created_at: str
    completed_at: Optional[str] = None


class TestHistoryResponse(BaseModel):
    sessions: List[TestSessionResponse]
    total: int


class CompleteTestRequest(BaseModel):
    user_id: str
    course_id: str
    test_id: str


class CompleteTestResponse(BaseModel):
    test_id: str
    status: str
    pass_chance: Optional[float] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{user_id}/{course_id}", response_model=TestResponse)
async def get_test(user_id: str, course_id: str):
    """
    Generate a test for a course. Returns one question per concept,
    avoiding recently-answered questions where possible.
    """
    try:
        result = await generate_test(user_id, course_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/resume/{user_id}/{session_id}", response_model=TestResponse)
async def resume_test_endpoint(user_id: str, session_id: str):
    """
    Resume an in-progress test session. Returns the full question set
    plus counts of already-answered questions.
    """
    try:
        result = await resume_test(user_id, session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/answer", response_model=AnswerResponse)
async def submit_answer(req: AnswerRequest):
    """
    Submit a single answer. Returns correctness feedback + BKT mastery update.
    """
    try:
        result = await process_test_answer(
            user_id=req.user_id,
            course_id=req.course_id,
            question_id=req.question_id,
            selected_option_index=req.selected_option_index,
            test_id=req.test_id,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/pass-chance/{user_id}/{course_id}", response_model=PassChanceResponse)
async def get_pass_chance_endpoint(user_id: str, course_id: str):
    """
    Get the current estimated passing chance for a user in a course.
    """
    try:
        result = await get_pass_chance(user_id, course_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/history/{user_id}/{course_id}", response_model=TestHistoryResponse)
async def get_test_history_endpoint(user_id: str, course_id: str):
    """
    Get quiz history for a user in a course (most recent first).
    """
    try:
        sessions = await get_test_history(user_id, course_id)
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/complete", response_model=CompleteTestResponse)
async def complete_test_endpoint(req: CompleteTestRequest):
    """
    Explicitly mark a test session as completed.
    Records the final pass chance.
    """
    try:
        result = await complete_test_session(
            user_id=req.user_id,
            course_id=req.course_id,
            test_id=req.test_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
