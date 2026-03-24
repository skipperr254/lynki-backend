from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.schemas.bkt import (
    BKTSummaryResponse,
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
    concept_ids: Optional[str] = Query(None, description="Comma-separated concept UUIDs for a targeted session"),
):
    """
    Get an adaptive study session for a course.
    Returns questions selected via weighted random from unmastered concepts.
    Priority: concept_ids > topic_id > whole course.
    """
    try:
        concept_id_list = (
            [c.strip() for c in concept_ids.split(",") if c.strip()]
            if concept_ids
            else None
        )
        result = await BKTService.get_next_session(
            user_id=user_id,
            course_id=course_id,
            topic_id=topic_id,
            concept_ids=concept_id_list,
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


@router.get("/mastery/{user_id}/{course_id}", response_model=BKTSummaryResponse)
async def get_mastery(user_id: str, course_id: str):
    """Get flat skill list + aggregate pass probability for a course."""
    try:
        return await BKTService.get_mastery_for_course(user_id, course_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
