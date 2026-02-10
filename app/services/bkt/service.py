from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.async_db import db_select_single
from app.services.bkt.math import BKTParams, soft_evidence_update
from app.services.bkt.repository import BKTRepository


def _normalize_score(claude_score: float) -> float:
    try:
        x = float(claude_score) / 100.0
    except Exception:
        x = 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class BKTService:
    """Orchestration layer for updating BKT mastery.

    Milestone 2 scope:
      - Correct soft-evidence BKT math
      - Persistence into bkt_mastery
      - Resolve KC(s) for a question (currently concept-based)
    """

    @staticmethod
    async def _get_kc_ids_for_question(question_id: str) -> List[str]:
        """Resolve Knowledge Component IDs for the question.

        Current Lynki schema is concept-based (questions.concept_id).
        For multi-KC later: create a join table and return multiple IDs.
        """
        qrow = await db_select_single("questions", filters={"id": question_id})
        if not qrow:
            raise ValueError(f"Question not found: {question_id}")

        kc_id = qrow.get("concept_id") or qrow.get("knowledge_component_id")
        if not kc_id:
            raise ValueError(f"Question {question_id} has no concept_id/knowledge_component_id mapping")

        return [str(kc_id)]

    @staticmethod
    async def update_mastery_for_response(
        user_id: str,
        question_id: str,
        claude_score: float,
        kc_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Update mastery for all KCs linked to the question."""
        q = _normalize_score(claude_score)
        kc_ids = await BKTService._get_kc_ids_for_question(question_id)

        results: List[Dict[str, Any]] = []

        for kc_id in kc_ids:
            row = await BKTRepository.get_row(user_id, kc_id)
            if not row:
                row = await BKTRepository.create_row(user_id, kc_id)

            p_before = float(row["p_mastery"])
            params = BKTParams(
                p_learn=float(row["p_learn"]),
                p_guess=float(row["p_guess"]),
                p_slip=float(row["p_slip"]),
            )

            # Optional weighting (conservative blend with 0.5 = uncertain)
            q_eff = q
            if kc_weights and kc_id in kc_weights:
                w = float(kc_weights[kc_id])
                w = 0.0 if w < 0.0 else 1.0 if w > 1.0 else w
                q_eff = (1.0 - w) * 0.5 + w * q

            p_after, debug = soft_evidence_update(p_before, q_eff, params)

            updated = await BKTRepository.update_row(
                row_id=str(row["id"]),
                updates={
                    "p_mastery": p_after,
                    "total_attempts": int(row.get("total_attempts", 0)) + 1,
                },
            )

            results.append(
                {
                    "knowledge_component_id": kc_id,
                    "p_mastery_before": p_before,
                    "p_mastery_after": float(updated["p_mastery"]),
                    "debug": debug,
                }
            )

        return {"user_id": user_id, "question_id": question_id, "q": q, "updated": results}
