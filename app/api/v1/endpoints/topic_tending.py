"""
Topic Tending endpoints — FastAPI router for the Tending Flow feature.

Endpoint overview:
  POST /topic-tending/generate        → create session, generate study content
  POST /topic-tending/evaluate-recall → evaluate student's active-recall text
  POST /topic-tending/complete        → compute BKT mastery delta, mark session done

Auth: this app does not currently use JWT middleware on individual routes
(the service role key is used server-side for all Supabase calls).
User identity comes from the request body (same pattern as study_plan, topic_quiz).

Erik fills in the Claude prompt logic for /generate and /evaluate-recall via PR.
Peter owns /complete (BKT integration).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from app.core.supabase import get_supabase
from app.core.async_db import run_db_operation
from app.schemas.topic_tending import (
    CompleteRequest,
    CompleteResponse,
    EvaluateRecallRequest,
    EvaluateRecallResponse,
    GenerateRequest,
    GenerateResponsePayload,
    KCDelta,
)
from app.services.bkt.service import BKTService, DEFAULTS
from app.services.tending_service import TendingService

router = APIRouter()
logger = logging.getLogger(__name__)

_supabase = get_supabase()


# ─── Internal helpers ──────────────────────────────────────────────────────────


async def _get_topic_mastery(user_id: str, course_id: str, topic_id: str) -> float:
    """
    Average p_mastery across all KCs belonging to this topic.
    Returns a value in [0.0, 1.0].  Falls back to the BKT default (0.2) when
    the user has no mastery rows yet so new sessions don't show 0%.
    """
    concepts_resp = await run_db_operation(
        lambda: _supabase.table("concepts")
        .select("id")
        .eq("topic_id", topic_id)
        .execute()
    )
    concepts = getattr(concepts_resp, "data", None) or []
    if not concepts:
        return DEFAULTS["p_mastery"]

    kc_ids = [c["id"] for c in concepts]

    mastery_resp = await run_db_operation(
        lambda: _supabase.table("bkt_mastery")
        .select("knowledge_component_id, p_mastery")
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .in_("knowledge_component_id", kc_ids)
        .execute()
    )
    rows = getattr(mastery_resp, "data", None) or []

    if not rows:
        return DEFAULTS["p_mastery"]

    return sum(float(r["p_mastery"]) for r in rows) / len(rows)


async def _get_kc_breakdown_for_topic(
    user_id: str,
    course_id: str,
    topic_id: str,
    mastery_before_map: Optional[Dict[str, float]] = None,
) -> List[KCDelta]:
    """
    Build a per-KC breakdown of mastery before/after for the Mastery Delta screen.
    mastery_before_map is a snapshot taken BEFORE any BKT updates this session.
    If not supplied, before == after (no change recorded).
    """
    # Fetch concepts for this topic with their names
    concepts_resp = await run_db_operation(
        lambda: _supabase.table("concepts")
        .select("id, name")
        .eq("topic_id", topic_id)
        .execute()
    )
    concepts = getattr(concepts_resp, "data", None) or []
    if not concepts:
        return []

    kc_ids = [c["id"] for c in concepts]
    concept_names = {c["id"]: c["name"] for c in concepts}

    # Fetch current (after) mastery rows
    mastery_resp = await run_db_operation(
        lambda: _supabase.table("bkt_mastery")
        .select("knowledge_component_id, p_mastery")
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .in_("knowledge_component_id", kc_ids)
        .execute()
    )
    rows = getattr(mastery_resp, "data", None) or []
    after_map = {r["knowledge_component_id"]: float(r["p_mastery"]) for r in rows}

    breakdown: List[KCDelta] = []
    for kc_id in kc_ids:
        after = after_map.get(kc_id, DEFAULTS["p_mastery"])
        before = (mastery_before_map or {}).get(kc_id, after)
        breakdown.append(
            KCDelta(
                kc_id=kc_id,
                name=concept_names.get(kc_id, kc_id),
                before=round(before, 4),
                after=round(after, 4),
            )
        )

    return breakdown


# ─── POST /generate ───────────────────────────────────────────────────────────


@router.post("/generate", response_model=GenerateResponsePayload)
async def generate_tending_session(req: GenerateRequest):
    """
    Fetch BKT mastery + topic content, call Sonnet 4.6 to
    generate recall cards, mnemonics, active-recall prompt, concept pairs.
    Insert a row into topic_tending_sessions and return the generated payload.
    """
    try:
        service = TendingService()
        return await service.generate_session(req.user_id, req.course_id, req.topic_id)
    except Exception as e:
        logger.error(f"Error generating tending session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── POST /evaluate-recall ────────────────────────────────────────────────────


@router.post("/evaluate-recall", response_model=EvaluateRecallResponse)
async def evaluate_recall(req: EvaluateRecallRequest):
    """
    Look up the source_paragraph from generated_content,
    call Sonnet 4.6 to compare student_response against it.
    Return matched concepts (got_right) and missed concepts plus source_paragraph.
    Persist to active_recall_input and active_recall_evaluation columns.
    """
    try:
        service = TendingService()
        result = await service.evaluate_recall(req.session_id, req.student_response)

        # Persist to active_recall_input and active_recall_evaluation
        await run_db_operation(
            lambda: _supabase.table("topic_tending_sessions").update({
                "active_recall_input": req.student_response,
                "active_recall_evaluation": result
            }).eq("id", req.session_id).execute()
        )

        return result
    except Exception as e:
        logger.error(f"Error evaluating recall: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── POST /complete ───────────────────────────────────────────────────────────


@router.post("/complete", response_model=CompleteResponse)
async def complete_session(req: CompleteRequest):
    """
    Finalize a tending session:
      1. Load the session row (verify it exists and isn't already completed).
      2. Snapshot per-KC mastery BEFORE any updates.
      3. If the quiz ran (question_ids present in results.quiz), the BKT was
         already updated by the existing /topic-quiz/answer flow — no double-count.
      4. If the quiz was skipped/not done, apply a soft BKT nudge from recall
         card ratings and active-recall evaluation score.
      5. Read mastery_after (average across topic's KCs).
      6. Persist results + mark session complete.
      7. Return the full MasteryDelta that the frontend renders.
    """
    # ── 1. Load session ──────────────────────────────────────────────────────
    session_resp = await run_db_operation(
        lambda: _supabase.table("topic_tending_sessions")
        .select("*")
        .eq("id", req.session_id)
        .maybe_single()
        .execute()
    )
    session = getattr(session_resp, "data", None)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("completed_at"):
        raise HTTPException(status_code=400, detail="Session already completed")

    course_id: str = session["course_id"]
    topic_id: str = session["topic_id"]
    user_id: str = session["user_id"]

    # ── 2. Snapshot mastery BEFORE any BKT updates ───────────────────────────
    #   We read per-KC rows now so the breakdown can show a meaningful "before".
    concepts_resp = await run_db_operation(
        lambda: _supabase.table("concepts")
        .select("id, name")
        .eq("topic_id", topic_id)
        .execute()
    )
    concepts = getattr(concepts_resp, "data", None) or []
    kc_ids = [c["id"] for c in concepts]

    if kc_ids:
        pre_mastery_resp = await run_db_operation(
            lambda: _supabase.table("bkt_mastery")
            .select("knowledge_component_id, p_mastery")
            .eq("user_id", user_id)
            .eq("course_id", course_id)
            .in_("knowledge_component_id", kc_ids)
            .execute()
        )
        pre_rows = getattr(pre_mastery_resp, "data", None) or []
    else:
        pre_rows = []

    mastery_before_map: Dict[str, float] = {
        r["knowledge_component_id"]: float(r["p_mastery"]) for r in pre_rows
    }

    mastery_before = (
        sum(mastery_before_map.values()) / len(mastery_before_map)
        if mastery_before_map
        else DEFAULTS["p_mastery"]
    )

    # ── 3 + 4. BKT update logic ───────────────────────────────────────────────
    quiz_did_run = bool(
        req.results.quiz
        and req.results.quiz.question_ids
        and len(req.results.quiz.question_ids) > 0
    )

    if quiz_did_run:
        # Quiz answers were submitted through the existing /topic-quiz/answer
        # flow which already called BKTService.update_mastery_for_concept for
        # each answered question.  Don't double-count — just persist the link.
        logger.info(
            "complete_session: quiz ran for session=%s (%d questions). "
            "BKT already updated via quiz flow.",
            req.session_id,
            len(req.results.quiz.question_ids),  # type: ignore[union-attr]
        )
    else:
        # Quiz was skipped or not reached. Apply a soft nudge from:
        #   a) Recall card self-ratings  (got_it = strong positive signal)
        #   b) Active-recall evaluation  (score reflects written recall quality)
        if kc_ids:
            await BKTService.apply_soft_evidence(
                user_id=user_id,
                course_id=course_id,
                topic_id=topic_id,
                kc_ids=kc_ids,
                recall_results=[r.model_dump() for r in (req.results.recall or [])],
                active_recall_evaluation=(
                    req.results.active_recall.evaluation
                    if req.results.active_recall
                    else None
                ),
            )

    # ── 5. Read mastery_after ─────────────────────────────────────────────────
    mastery_after = await _get_topic_mastery(user_id, course_id, topic_id)

    # ── 6. Persist results + mark complete ───────────────────────────────────
    update_payload: Dict[str, Any] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "mastery_before": mastery_before,
        "mastery_after": mastery_after,
        "current_step": "complete",
        "stages_skipped": [s for s in req.results.stages_skipped],
    }

    # Persist recall card results
    if req.results.recall:
        update_payload["recall_card_results"] = [
            {"id": r.id, "got_it": r.rating == "got_it"} for r in req.results.recall
        ]

    # Persist active recall input
    if req.results.active_recall:
        update_payload["active_recall_input"] = req.results.active_recall.student_response
        if req.results.active_recall.evaluation:
            update_payload["active_recall_evaluation"] = req.results.active_recall.evaluation

    # Persist concept pair results
    if req.results.connections:
        correct = sum(1 for c in req.results.connections if c.matched)
        incorrect = sum(1 for c in req.results.connections if not c.matched)
        update_payload["concept_pair_results"] = {
            "correct": correct,
            "incorrect": incorrect,
        }

    await run_db_operation(
        lambda: _supabase.table("topic_tending_sessions")
        .update(update_payload)
        .eq("id", req.session_id)
        .execute()
    )

    # ── 7. Build + return MasteryDelta ────────────────────────────────────────
    kc_breakdown = await _get_kc_breakdown_for_topic(
        user_id, course_id, topic_id, mastery_before_map
    )

    # Fetch topic name for the response
    topic_resp = await run_db_operation(
        lambda: _supabase.table("topics")
        .select("name")
        .eq("id", topic_id)
        .maybe_single()
        .execute()
    )
    topic_data = getattr(topic_resp, "data", None) or {}
    topic_title = topic_data.get("name", "")

    return CompleteResponse(
        stage="mastery_delta",
        topic_title=topic_title,
        mastery_before=round(mastery_before, 4),
        mastery_after=round(mastery_after, 4),
        kc_breakdown=kc_breakdown,
        tended_today=True,
    )
