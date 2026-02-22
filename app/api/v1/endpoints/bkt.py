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
# New adaptive learning endpoints
# ---------------------------------------------------------------------------

@router.get("/session/{user_id}/{document_id}", response_model=BKTSessionResponse)
async def get_session(
    user_id: str,
    document_id: str,
    topic_id: Optional[str] = Query(None, description="Scope session to a specific topic"),
):
    """
    Get an adaptive study session.
    Returns questions selected via weighted random from unmastered concepts.
    Questions do NOT include correct answers — answers are validated server-side.
    """
    try:
        result = await BKTService.get_next_session(
            user_id=user_id,
            document_id=document_id,
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
            document_id=req.document_id,
            selected_option_index=req.selected_option_index,
            session_id=req.session_id,
            time_spent_ms=req.time_spent_ms,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/progress/{user_id}/{document_id}", response_model=BKTProgressResponse)
async def get_progress(user_id: str, document_id: str):
    """
    Get full document progress tree: topics -> concepts with BKT mastery values.
    """
    try:
        return await BKTService.get_document_progress(user_id, document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Legacy endpoints (kept for backward compat)
# ---------------------------------------------------------------------------

@router.post("/update", response_model=BKTUpdateResponse)
async def update_bkt(req: BKTUpdateRequest):
    try:
        return await BKTService.update_mastery_for_response(
            user_id=req.user_id,
            question_id=req.question_id,
            document_id=req.document_id,
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
            document_id=req.document_id,
            updates=updates,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/mastery/{user_id}/{document_id}", response_model=BKTSummaryResponse)
async def get_mastery(user_id: str, document_id: str):
    try:
        return await BKTService.get_mastery_for_document(user_id, document_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/weak-skills/{user_id}/{document_id}", response_model=BKTWeakSkillsResponse)
async def get_weak_skills(user_id: str, document_id: str):
    try:
        return await BKTService.get_weak_skills_for_document(user_id, document_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
