from __future__ import annotations

from typing import Any, Dict, Optional

from app.core.async_db import db_select_single, db_insert, db_update

DEFAULTS: Dict[str, Any] = {
    "p_mastery": 0.3,
    "p_learn": 0.1,
    "p_slip": 0.1,
    "p_guess": 0.25,
}


class BKTRepository:
    """DB access layer for the bkt_mastery table."""

    @staticmethod
    async def get_row(user_id: str, kc_id: str) -> Optional[Dict[str, Any]]:
        return await db_select_single(
            "bkt_mastery",
            filters={"user_id": user_id, "knowledge_component_id": kc_id},
        )

    @staticmethod
    async def create_row(user_id: str, kc_id: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "user_id": user_id,
            "knowledge_component_id": kc_id,
            **DEFAULTS,
            "total_attempts": 0,
        }
        if overrides:
            payload.update(overrides)

        inserted = await db_insert("bkt_mastery", payload)
        return inserted[0] if isinstance(inserted, list) and inserted else inserted

    @staticmethod
    async def update_row(row_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        updated = await db_update("bkt_mastery", updates, filters={"id": row_id})
        return updated[0] if isinstance(updated, list) and updated else updated
