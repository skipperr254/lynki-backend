from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.schemas.bkt import (
    BKTUpdateRequest,
    BKTUpdateResponse,
    BKTSummaryResponse,
    BKTWeakSkillsResponse,
    BKTBatchUpdateRequest,
    BKTBatchUpdateResponse,
    BKTSessionResponse,
    BKTAnswerRequest,
    BKTAnswerResponse,
    BKTProgressResponse,
)
from app.services.bkt.service import BKTService

router = APIRouter()


# ---------------------------------------------------------------------------
# Adaptive learning endpoints (course-scoped)
# ---------------------------------------------------------------------------

@router.get("/session/{user_id}/{course_id}", response_model=BKTSessionResponse)
async def get_session(
    user_id: str,
    course_id: str,
    topic_id: Optional[str] = Query(None, description="Scope session to a specific topic"),
):
    """
    Get an adaptive study session for a course.
    Returns questions selected via weighted random from unmastered concepts
    across all documents in the course.
    Questions do NOT include correct answers — answers are validated server-side.
    """
    try:
        result = await BKTService.get_next_session(
            user_id=user_id,
            course_id=course_id,
            topic_id=topic_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/answer", response_model=BKTAnswerResponse)
async def submit_answer(req: BKTAnswerRequest):
    """
    Submit a single answer.
    Checks correctness, runs BKT update, records the attempt, returns feedback.
    """
    try:
        result = await BKTService.process_answer(
            user_id=req.user_id,
            question_id=req.question_id,
            course_id=req.course_id,
            selected_option_index=req.selected_option_index,
            session_id=req.session_id,
            time_spent_ms=req.time_spent_ms,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/progress/{user_id}/{course_id}", response_model=BKTProgressResponse)
async def get_progress(user_id: str, course_id: str):
    """
    Get full course progress tree: topics -> concepts with BKT mastery values.
    Aggregates across all documents in the course.
    """
    try:
        return await BKTService.get_course_progress(user_id, course_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Legacy-style endpoints (now course-scoped)
# ---------------------------------------------------------------------------

@router.post("/update", response_model=BKTUpdateResponse)
async def update_bkt(req: BKTUpdateRequest):
    try:
        return await BKTService.update_mastery_for_response(
            user_id=req.user_id,
            question_id=req.question_id,
            course_id=req.course_id,
            claude_score=req.claude_score,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/update-batch", response_model=BKTBatchUpdateResponse)
async def update_bkt_batch(req: BKTBatchUpdateRequest):
    try:
        updates = [(u.question_id, u.claude_score) for u in req.updates]
        return await BKTService.update_mastery_batch(
            user_id=req.user_id,
            course_id=req.course_id,
            updates=updates,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/mastery/{user_id}/{course_id}", response_model=BKTSummaryResponse)
async def get_mastery(user_id: str, course_id: str):
    try:
        return await BKTService.get_mastery_for_course(user_id, course_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/weak-skills/{user_id}/{course_id}", response_model=BKTWeakSkillsResponse)
async def get_weak_skills(user_id: str, course_id: str):
    try:
        return await BKTService.get_weak_skills_for_course(user_id, course_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
