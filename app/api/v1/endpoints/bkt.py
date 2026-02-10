from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.bkt import (
    BKTUpdateRequest,
    BKTUpdateResponse,
    BKTSummaryResponse,
    BKTWeakSkillsResponse,
)
from app.services.bkt.service import BKTService

router = APIRouter()


@router.post("/update", response_model=BKTUpdateResponse)
async def update_bkt(req: BKTUpdateRequest):
    try:
        result = await BKTService.update_mastery_for_response(
            user_id=req.user_id,
            question_id=req.question_id,
            claude_score=req.claude_score,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/mastery/{user_id}/{document_id}", response_model=BKTSummaryResponse)
async def get_mastery(user_id: str, document_id: str):
    """Return aggregated pass probability and per-skill mastery.

    NOTE: For now, we treat document_id as the 'subject' grouping,
    since the current Lynki schema is document/topic/concept-based.
    """
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
