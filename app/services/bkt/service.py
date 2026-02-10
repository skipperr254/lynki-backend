from __future__ import annotations

from math import prod
from typing import Any, Dict, List, Optional

from app.core.supabase import get_supabase
from app.core.async_db import db_select, db_select_single, db_insert, db_update
from app.services.bkt.math import BKTParams, soft_evidence_update

# Single Supabase client (Service Role) for backend system operations
_supabase = get_supabase()

DEFAULTS: Dict[str, Any] = {
    "p_mastery": 0.3,  # P(L0)
    "p_learn": 0.1,    # P(T)
    "p_slip": 0.1,     # P(S)
    "p_guess": 0.25,   # P(G)
    "total_attempts": 0,
}


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _normalize_score(claude_score: float) -> float:
    """
    Normalize Claude score 0-100 to q in [0,1].
    For binary MCQ systems, caller can pass 0 or 100.
    """
    try:
        return _clamp01(float(claude_score) / 100.0)
    except Exception:
        return 0.0


def aggregate_pass_probability(p_list: List[float]) -> float:
    """Noisy-OR aggregation: 1 - Π(1 - p_k)."""
    if not p_list:
        return 0.0
    return 1.0 - prod([1.0 - _clamp01(float(p)) for p in p_list])


async def _select(table: str, columns: str = "*", **filters) -> List[Dict[str, Any]]:
    resp = await db_select(_supabase, table, columns, **filters)
    data = getattr(resp, "data", None)
    return data or []


async def _select_single(table: str, columns: str = "*", **filters) -> Optional[Dict[str, Any]]:
    resp = await db_select_single(_supabase, table, columns, **filters)
    data = getattr(resp, "data", None)
    return data or None


async def _insert(table: str, data: dict | list) -> Any:
    resp = await db_insert(_supabase, table, data)
    return getattr(resp, "data", None)


async def _update(table: str, data: dict, **filters) -> Any:
    resp = await db_update(_supabase, table, data, **filters)
    return getattr(resp, "data", None)


class BKTService:
    """
    BKT orchestration layer.

    - Items: questions
    - KCs: concepts (current schema uses questions.concept_id and concepts.document_id)
    - State: bkt_mastery per (user_id, knowledge_component_id)
    """

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

        # MVP: single-KC. Multi-KC can later return multiple IDs.
        return [str(kc_id)]

    @staticmethod
    async def _get_or_create_mastery_row(user_id: str, kc_id: str) -> Dict[str, Any]:
        row = await _select_single(
            "bkt_mastery",
            "*",
            user_id=user_id,
            knowledge_component_id=kc_id,
        )
        if row:
            return row

        payload = {
            "user_id": user_id,
            "knowledge_component_id": kc_id,
            **DEFAULTS,
        }
        inserted = await _insert("bkt_mastery", payload)
        if isinstance(inserted, list) and inserted:
            return inserted[0]
        if isinstance(inserted, dict):
            return inserted

        # Fallback: re-read
        row = await _select_single(
            "bkt_mastery",
            "*",
            user_id=user_id,
            knowledge_component_id=kc_id,
        )
        if not row:
            raise RuntimeError("Failed to create bkt_mastery row")
        return row

    @staticmethod
    async def update_mastery_for_response(
        user_id: str,
        question_id: str,
        claude_score: float,
        kc_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Update mastery for all KCs linked to the question.

        claude_score: 0-100 (soft evidence). For binary MCQ:
          - correct => 100
          - incorrect => 0
        """
        q = _normalize_score(claude_score)
        kc_ids = await BKTService._get_kc_ids_for_question(question_id)

        results: List[Dict[str, Any]] = []

        for kc_id in kc_ids:
            row = await BKTService._get_or_create_mastery_row(user_id, kc_id)

            p_before = float(row.get("p_mastery", DEFAULTS["p_mastery"]))
            params = BKTParams(
                p_learn=float(row.get("p_learn", DEFAULTS["p_learn"])),
                p_guess=float(row.get("p_guess", DEFAULTS["p_guess"])),
                p_slip=float(row.get("p_slip", DEFAULTS["p_slip"])),
            )

            # Optional weighting for multi-KC: blend q towards 0.5 if low relevance.
            q_eff = q
            if kc_weights and kc_id in kc_weights:
                w = _clamp01(float(kc_weights[kc_id]))
                q_eff = (1.0 - w) * 0.5 + w * q

            p_after, debug = soft_evidence_update(p_before, q_eff, params)

            new_attempts = int(row.get("total_attempts") or 0) + 1
            updated = await _update(
                "bkt_mastery",
                {"p_mastery": p_after, "total_attempts": new_attempts},
                id=str(row["id"]),
            )

            # Supabase update returns list
            if isinstance(updated, list) and updated:
                updated_row = updated[0]
            elif isinstance(updated, dict):
                updated_row = updated
            else:
                # fallback: re-read
                updated_row = await _select_single("bkt_mastery", "*", id=str(row["id"])) or {}

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
    async def get_mastery_for_document(user_id: str, document_id: str) -> Dict[str, Any]:
        """
        Aggregate mastery for all concepts in a document.
        MVP decision: document_id is the grouping scope.
        """
        concepts = await _select("concepts", "*", document_id=document_id)

        skills: List[Dict[str, Any]] = []
        p_list: List[float] = []

        for c in concepts:
            kc_id = str(c["id"])

            row = await _select_single(
                "bkt_mastery",
                "*",
                user_id=user_id,
                knowledge_component_id=kc_id,
            )

            # If no mastery row yet, treat as default prior (no DB write).
            if not row:
                p = float(DEFAULTS["p_mastery"])
                attempts = 0
            else:
                p = float(row.get("p_mastery", DEFAULTS["p_mastery"]))
                attempts = int(row.get("total_attempts") or 0)

            p_list.append(p)

            skills.append(
                {
                    "skill_name": c.get("title") or c.get("name") or c.get("skill_name") or kc_id,
                    "mastery": p,
                    "attempts": attempts,
                }
            )

        return {
            "pass_probability": aggregate_pass_probability(p_list),
            "skills": skills,
        }

    @staticmethod
    async def get_weak_skills_for_document(
        user_id: str, document_id: str, limit: int = 5
    ) -> Dict[str, Any]:
        mastery = await BKTService.get_mastery_for_document(user_id, document_id)
        skills = mastery["skills"]
        weakest = sorted(skills, key=lambda s: float(s["mastery"]))[:limit]
        return {"skills": weakest}
