"""
Profile endpoints — user settings (curriculum, etc.).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.core.supabase import get_supabase
from app.core.async_db import run_db_operation

router = APIRouter()
_supabase = get_supabase()

VALID_CURRICULA = {"percentage", "ib", "ap", "gcse", "a-level"}


class ProfileResponse(BaseModel):
    curriculum: str = "percentage"


class ProfileUpdateRequest(BaseModel):
    curriculum: str


@router.get("/{user_id}", response_model=ProfileResponse)
async def get_profile(user_id: str):
    """Get a user's profile settings."""
    try:
        resp = await run_db_operation(
            lambda: _supabase.table("user_profiles")
            .select("curriculum")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        data = getattr(resp, "data", None)
        if data:
            return ProfileResponse(curriculum=data.get("curriculum", "percentage"))
        return ProfileResponse(curriculum="percentage")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{user_id}", response_model=ProfileResponse)
async def update_profile(user_id: str, body: ProfileUpdateRequest):
    """Create or update a user's profile settings."""
    if body.curriculum not in VALID_CURRICULA:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid curriculum '{body.curriculum}'. Must be one of: {', '.join(sorted(VALID_CURRICULA))}",
        )

    try:
        resp = await run_db_operation(
            lambda: _supabase.table("user_profiles")
            .upsert(
                {"user_id": user_id, "curriculum": body.curriculum},
                on_conflict="user_id",
            )
            .execute()
        )
        data = getattr(resp, "data", None)
        if data and len(data) > 0:
            return ProfileResponse(curriculum=data[0].get("curriculum", body.curriculum))
        return ProfileResponse(curriculum=body.curriculum)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
