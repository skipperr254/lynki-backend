from __future__ import annotations

from typing import Any, Dict, Optional

from app.core.supabase import get_supabase
from app.core.async_db import db_select_single, db_insert, db_update

supabase = get_supabase()

class BKTRepository:
    @staticmethod
    async def get_row(user_id: str, kc_id: str):
        resp = await db_select_single(
            supabase,
            "bkt_mastery",
            "*",
            user_id=user_id,
            knowledge_component_id=kc_id,
        )
        return resp.data  # dict o None

    @staticmethod
    async def create_row(user_id: str, kc_id: str, overrides=None):
        payload = {
            "user_id": user_id,
            "knowledge_component_id": kc_id,
            "p_mastery": 0.3,
            "p_learn": 0.1,
            "p_slip": 0.1,
            "p_guess": 0.25,
            "total_attempts": 0,
            **(overrides or {}),
        }
        resp = await db_insert(supabase, "bkt_mastery", payload)
        # Supabase devuelve lista en resp.data
        return (resp.data[0] if resp.data else None)

    @staticmethod
    async def update_row(row_id: str, updates: dict):
        resp = await db_update(supabase, "bkt_mastery", updates, id=row_id)
        return (resp.data[0] if resp.data else None)
