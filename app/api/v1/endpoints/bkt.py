from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.bkt import (
    BKTUpdateRequest,
    BKTUpdateResponse,
    BKTSummaryResponse,
    BKTWeakSkillsResponse,
    BKTBatchUpdateRequest,
    BKTBatchUpdateResponse,
)
from app.services.bkt.service import BKTService

router = APIRouter()


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
