from __future__ import annotations

from math import prod
from typing import Any, Dict, List, Optional, Tuple

from app.core.supabase import get_supabase
from app.core.async_db import db_select, db_select_single, db_update, db_upsert
from app.services.bkt.math import BKTParams, soft_evidence_update

_supabase = get_supabase()

DEFAULTS: Dict[str, Any] = {
    "p_mastery": 0.3,
    "p_init": 0.3,
    "p_transit": 0.1,
    "p_guess": 0.25,
    "p_slip": 0.1,
    "n_attempts": 0,
    "n_correct": 0,
}


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _normalize_score(claude_score: float) -> float:
    try:
        return _clamp01(float(claude_score) / 100.0)
    except Exception:
        return 0.0


def aggregate_pass_probability(p_list: List[float]) -> float:
    if not p_list:
        return 0.0
    return 1.0 - prod([1.0 - _clamp01(float(p)) for p in p_list])


async def _select(table: str, columns: str = "*", **filters) -> List[Dict[str, Any]]:
    resp = await db_select(_supabase, table, columns, **filters)
    return getattr(resp, "data", None) or []


async def _select_single(table: str, columns: str = "*", **filters) -> Optional[Dict[str, Any]]:
    resp = await db_select_single(_supabase, table, columns, **filters)
    return getattr(resp, "data", None) or None


async def _update(table: str, data: dict, **filters) -> Any:
    resp = await db_update(_supabase, table, data, **filters)
    return getattr(resp, "data", None)


async def _upsert(table: str, data: dict | list, on_conflict: str) -> Any:
    resp = await db_upsert(_supabase, table, data, on_conflict=on_conflict)
    return getattr(resp, "data", None)


class BKTService:
    @staticmethod
    async def _get_kc_ids_for_question(question_id: str) -> List[str]:
        qrow = await _select_single("questions", "*", id=question_id)
        if not qrow:
            raise ValueError(f"Question not found: {question_id}")

        kc_id = qrow.get("concept_id") or qrow.get("knowledge_component_id")
        if not kc_id:
            raise ValueError(
                f"Question {question_id} has no concept_id/knowledge_component_id mapping"
            )
        return [str(kc_id)]

    @staticmethod
    async def _ensure_mastery_row(user_id: str, document_id: str, kc_id: str) -> Dict[str, Any]:
        """
        Concurrency-safe row creation:
        - upsert on unique key (user_id, document_id, knowledge_component_id)
        """
        payload = {
            "user_id": user_id,
            "document_id": document_id,
            "knowledge_component_id": kc_id,
        }

        # One statement that either inserts or returns existing row (depending on PostgREST behavior).
        # If it returns empty in some setups, we re-read.
        data = await _upsert(
            "bkt_mastery",
            payload,
            on_conflict="user_id,document_id,knowledge_component_id",
        )

        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict) and data:
            return data

        # Fallback: re-read as list (safe)
        rows = await _select(
            "bkt_mastery",
            "*",
            user_id=user_id,
            document_id=document_id,
            knowledge_component_id=kc_id,
        )
        if not rows:
            raise RuntimeError("Failed to ensure bkt_mastery row")
        return rows[0]

    @staticmethod
    async def update_mastery_for_response(
        user_id: str,
        question_id: str,
        document_id: str,
        claude_score: float,
        kc_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        q = _normalize_score(claude_score)
        kc_ids = await BKTService._get_kc_ids_for_question(question_id)

        results: List[Dict[str, Any]] = []

        for kc_id in kc_ids:
            row = await BKTService._ensure_mastery_row(user_id, document_id, kc_id)

            p_before = float(row.get("p_mastery", DEFAULTS["p_mastery"]))
            params = BKTParams(
                p_learn=float(row.get("p_transit", DEFAULTS["p_transit"])),
                p_guess=float(row.get("p_guess", DEFAULTS["p_guess"])),
                p_slip=float(row.get("p_slip", DEFAULTS["p_slip"])),
            )

            q_eff = q
            if kc_weights and kc_id in kc_weights:
                w = _clamp01(float(kc_weights[kc_id]))
                q_eff = (1.0 - w) * 0.5 + w * q

            p_after, debug = soft_evidence_update(p_before, q_eff, params)

            new_attempts = int(row.get("n_attempts") or 0) + 1
            new_correct = int(row.get("n_correct") or 0) + (1 if q_eff >= 0.5 else 0)

            updated_data = await _update(
                "bkt_mastery",
                {
                    "p_mastery": p_after,
                    "n_attempts": new_attempts,
                    "n_correct": new_correct,
                },
                id=str(row["id"]),
            )

            updated_row = {}
            if isinstance(updated_data, list) and updated_data:
                updated_row = updated_data[0]
            elif isinstance(updated_data, dict) and updated_data:
                updated_row = updated_data
            else:
                # fallback read
                reread = await _select("bkt_mastery", "*", id=str(row["id"]))
                updated_row = reread[0] if reread else {}

            results.append(
                {
                    "knowledge_component_id": kc_id,
                    "p_mastery_before": p_before,
                    "p_mastery_after": float(updated_row.get("p_mastery", p_after)),
                    "q_used": q_eff,
                    "debug": debug,
                }
            )

        return {"user_id": user_id, "question_id": question_id, "q": q, "updated": results}

    @staticmethod
    async def update_mastery_batch(
        user_id: str,
        document_id: str,
        updates: List[Tuple[str, float]],
    ) -> Dict[str, Any]:
        """
        updates: list of (question_id, claude_score)
        """
        out = []
        for (question_id, score) in updates:
            out.append(
                await BKTService.update_mastery_for_response(
                    user_id=user_id,
                    question_id=question_id,
                    document_id=document_id,
                    claude_score=score,
                )
            )
        return {"user_id": user_id, "document_id": document_id, "count": len(updates), "results": out}

    @staticmethod
    async def get_mastery_for_document(user_id: str, document_id: str) -> Dict[str, Any]:
        concepts = await _select("concepts", "*", document_id=document_id)

        skills: List[Dict[str, Any]] = []
        p_list: List[float] = []

        for c in concepts:
            kc_id = str(c["id"])
            rows = await _select(
                "bkt_mastery",
                "*",
                user_id=user_id,
                document_id=document_id,
                knowledge_component_id=kc_id,
            )
            row = rows[0] if rows else None

            if not row:
                p = float(DEFAULTS["p_mastery"])
                attempts = 0
            else:
                p = float(row.get("p_mastery", DEFAULTS["p_mastery"]))
                attempts = int(row.get("n_attempts") or 0)

            p_list.append(p)
            skills.append(
                {
                    "skill_name": c.get("title") or c.get("name") or c.get("skill_name") or kc_id,
                    "mastery": p,
                    "attempts": attempts,
                }
            )

        return {"pass_probability": aggregate_pass_probability(p_list), "skills": skills}

    @staticmethod
    async def get_weak_skills_for_document(user_id: str, document_id: str, limit: int = 5) -> Dict[str, Any]:
        mastery = await BKTService.get_mastery_for_document(user_id, document_id)
        weakest = sorted(mastery["skills"], key=lambda s: float(s["mastery"]))[:limit]
        return {"skills": weakest}
