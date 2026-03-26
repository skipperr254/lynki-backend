"""
BKT (Bayesian Knowledge Tracing) Service — adaptive learning engine.

This service is the single source of truth for mastery tracking.
It drives:
  - Mastery updates after each answer (Bayesian posterior + learning transition)
  - Adaptive session generation (weighted random concept selection, interleaved questions)
  - Progress reporting (per-concept, per-topic, per-course mastery)

The frontend is a thin client: it fetches sessions, submits answers, and displays progress.
All question-selection intelligence lives here.

Mastery is scoped per COURSE. A course contains multiple documents, each with topics
and concepts. All concepts across all documents in a course share one mastery pool.
"""

from __future__ import annotations

import logging
import math
import random
import uuid
from datetime import datetime, timedelta, timezone
from math import prod
from typing import Any, Dict, List, Optional, Tuple

from app.core.supabase import get_supabase
from app.core.async_db import (
    db_insert,
    db_select,
    db_select_single,
    db_update,
    db_upsert,
    run_db_operation,
)
from app.services.bkt.math import BKTParams, soft_evidence_update

logger = logging.getLogger(__name__)

_supabase = get_supabase()

# ---------------------------------------------------------------------------
# BKT configuration constants
# ---------------------------------------------------------------------------

MASTERY_THRESHOLD: float = 0.85
SESSION_MAX_QUESTIONS: int = 12
SESSION_MIN_QUESTIONS: int = 5
RECENTLY_SEEN_HOURS: int = 4
MAX_CONCEPTS_PER_SESSION: int = 5

DEFAULTS: Dict[str, Any] = {
    "p_mastery": 0.2,
    "p_init": 0.2,
    "p_transit": 0.15,
    "p_guess": 0.2,
    "p_slip": 0.1,
    "n_attempts": 0,
    "n_correct": 0,
}


# ---------------------------------------------------------------------------
# Low-level DB helpers
# ---------------------------------------------------------------------------

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


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def aggregate_pass_probability(p_list: List[float], target: float = 1.0) -> float:
    """
    Probability of scoring >= target on a test covering all concepts.

    Models each concept as a Bernoulli trial with probability p_i of answering
    correctly.  The fraction-correct score has:
        mu    = mean(p_i)
        sigma = sqrt( sum(p_i*(1-p_i)) / N^2 )

    Uses normal approximation:  P(score >= target) = 1 - Phi((target - mu) / sigma)
    """
    if not p_list:
        return 0.0

    n = len(p_list)
    clamped = [_clamp01(float(p)) for p in p_list]
    mu = sum(clamped) / n
    variance = sum(p * (1.0 - p) for p in clamped) / (n * n)
    sigma = math.sqrt(variance) if variance > 0 else 0.0

    target = _clamp01(target)

    if sigma == 0.0:
        return 1.0 if mu >= target else 0.0

    z = (target - mu) / sigma
    return _clamp01(1.0 - _normal_cdf(z))


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


async def _insert(table: str, data: dict | list) -> Any:
    resp = await db_insert(_supabase, table, data)
    return getattr(resp, "data", None)


# ---------------------------------------------------------------------------
# Course-scoped data fetchers
# ---------------------------------------------------------------------------

async def _get_documents_for_course(course_id: str) -> List[Dict[str, Any]]:
    """Fetch all documents belonging to a course."""
    return await _select("documents", "id, title", course_id=course_id)


async def _get_topics_for_course(course_id: str) -> List[Dict[str, Any]]:
    """Fetch all topics across all documents in a course."""
    docs = await _get_documents_for_course(course_id)
    if not docs:
        return []
    doc_ids = [d["id"] for d in docs]
    resp = await run_db_operation(
        lambda: _supabase.table("topics")
        .select("id, name, document_id")
        .in_("document_id", doc_ids)
        .execute()
    )
    return getattr(resp, "data", None) or []


async def _get_concepts_for_course(course_id: str) -> List[Dict[str, Any]]:
    """Fetch all concepts across all documents in a course (via documents -> topics -> concepts)."""
    topics = await _get_topics_for_course(course_id)
    if not topics:
        return []
    topic_ids = [t["id"] for t in topics]
    resp = await run_db_operation(
        lambda: _supabase.table("concepts")
        .select("id, name, explanation, complexity_level, topic_id")
        .in_("topic_id", topic_ids)
        .execute()
    )
    return getattr(resp, "data", None) or []


async def _get_questions_for_concepts(concept_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch questions + options for a set of concept IDs."""
    if not concept_ids:
        return []
    resp = await run_db_operation(
        lambda: _supabase.table("questions")
        .select("id, question, hint, difficulty_level, concept_id, question_options(id, option_text, option_index, is_correct, explanation)")
        .in_("concept_id", concept_ids)
        .execute()
    )
    return getattr(resp, "data", None) or []


async def _get_recent_question_ids(
    user_id: str, concept_ids: List[str], hours: int = RECENTLY_SEEN_HOURS
) -> set:
    """Get question IDs the user has answered recently (within `hours`)."""
    if not concept_ids:
        return set()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    resp = await run_db_operation(
        lambda: _supabase.table("user_question_attempts")
        .select("question_id")
        .eq("user_id", user_id)
        .in_("concept_id", concept_ids)
        .gte("created_at", cutoff)
        .execute()
    )
    data = getattr(resp, "data", None) or []
    return {r["question_id"] for r in data}


class BKTService:
    """
    Core adaptive learning service.

    All mastery is scoped per COURSE. A course aggregates concepts from all
    its documents into a single mastery pool.

    Methods:
    - update_mastery_for_response / update_mastery_batch  — BKT updates after answers
    - get_next_session         — build an adaptive session (weighted random, interleaved)
    - process_answer           — check correctness + BKT update + record attempt
    - get_course_progress      — full mastery tree for the overview page
    - get_mastery_for_course   — flat skill list + pass probability
    - get_weak_skills_for_course — N weakest skills
    """

    # -----------------------------------------------------------------------
    # Mastery update helpers
    # -----------------------------------------------------------------------

    @staticmethod
    async def _get_kc_ids_for_question(question_id: str) -> List[str]:
        qrow = await _select_single("questions", "*", id=question_id)
        if not qrow:
            raise ValueError(f"Question not found: {question_id}")
        kc_id = qrow.get("concept_id") or qrow.get("knowledge_component_id")
        if not kc_id:
            raise ValueError(f"Question {question_id} has no concept mapping")
        return [str(kc_id)]

    @staticmethod
    async def _resolve_document_id(kc_id: str) -> Optional[str]:
        """Resolve concept → topic → document_id."""
        try:
            concept = await _select_single("concepts", "topic_id", id=kc_id)
            if concept and concept.get("topic_id"):
                topic = await _select_single("topics", "document_id", id=concept["topic_id"])
                if topic and topic.get("document_id"):
                    return str(topic["document_id"])
        except Exception:
            pass
        return None

    @staticmethod
    async def _ensure_mastery_row(user_id: str, course_id: str, kc_id: str) -> Dict[str, Any]:
        """Upsert on (user_id, course_id, knowledge_component_id)."""
        # Resolve document_id from concept chain (optional metadata)
        document_id = await BKTService._resolve_document_id(kc_id)

        payload = {
            "user_id": user_id,
            "course_id": course_id,
            "knowledge_component_id": kc_id,
        }
        if document_id:
            payload["document_id"] = document_id

        data = await _upsert(
            "bkt_mastery",
            payload,
            on_conflict="user_id,course_id,knowledge_component_id",
        )
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict) and data:
            return data
        rows = await _select(
            "bkt_mastery", "*",
            user_id=user_id, course_id=course_id, knowledge_component_id=kc_id,
        )
        if not rows:
            raise RuntimeError("Failed to ensure bkt_mastery row")
        return rows[0]

    @staticmethod
    async def update_mastery_for_concept(
        user_id: str,
        course_id: str,
        concept_id: str,
        is_correct: bool,
    ) -> Dict[str, Any]:
        """
        Update BKT mastery for a concept directly (no question_id needed).
        Used by topic quiz sessions whose questions are not in the questions table.
        """
        q = 1.0 if is_correct else 0.0
        row = await BKTService._ensure_mastery_row(user_id, course_id, concept_id)
        p_before = float(row.get("p_mastery", DEFAULTS["p_mastery"]))
        params = BKTParams(
            p_learn=float(row.get("p_transit", DEFAULTS["p_transit"])),
            p_guess=float(row.get("p_guess", DEFAULTS["p_guess"])),
            p_slip=float(row.get("p_slip", DEFAULTS["p_slip"])),
        )
        p_after, _ = soft_evidence_update(p_before, q, params)
        new_attempts = int(row.get("n_attempts") or 0) + 1
        new_correct = int(row.get("n_correct") or 0) + (1 if is_correct else 0)
        await _update(
            "bkt_mastery",
            {"p_mastery": p_after, "n_attempts": new_attempts, "n_correct": new_correct},
            id=str(row["id"]),
        )
        return {"p_mastery_before": p_before, "p_mastery_after": p_after}

    @staticmethod
    async def update_mastery_for_response(
        user_id: str,
        question_id: str,
        course_id: str,
        claude_score: float,
        kc_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        q = _normalize_score(claude_score)
        kc_ids = await BKTService._get_kc_ids_for_question(question_id)
        results: List[Dict[str, Any]] = []

        for kc_id in kc_ids:
            row = await BKTService._ensure_mastery_row(user_id, course_id, kc_id)
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
                {"p_mastery": p_after, "n_attempts": new_attempts, "n_correct": new_correct},
                id=str(row["id"]),
            )
            updated_row = {}
            if isinstance(updated_data, list) and updated_data:
                updated_row = updated_data[0]
            elif isinstance(updated_data, dict) and updated_data:
                updated_row = updated_data
            else:
                reread = await _select("bkt_mastery", "*", id=str(row["id"]))
                updated_row = reread[0] if reread else {}

            results.append({
                "knowledge_component_id": kc_id,
                "p_mastery_before": p_before,
                "p_mastery_after": float(updated_row.get("p_mastery", p_after)),
                "q_used": q_eff,
                "debug": debug,
            })

        return {"user_id": user_id, "question_id": question_id, "q": q, "updated": results}

    @staticmethod
    async def update_mastery_batch(
        user_id: str,
        course_id: str,
        updates: List[Tuple[str, float]],
    ) -> Dict[str, Any]:
        out = []
        for (question_id, score) in updates:
            out.append(
                await BKTService.update_mastery_for_response(
                    user_id=user_id, question_id=question_id,
                    course_id=course_id, claude_score=score,
                )
            )
        return {"user_id": user_id, "course_id": course_id, "count": len(updates), "results": out}

    # -----------------------------------------------------------------------
    # Adaptive session generation
    # -----------------------------------------------------------------------

    @staticmethod
    async def get_next_session(
        user_id: str,
        course_id: str,
        topic_id: Optional[str] = None,
        concept_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build an adaptive study session for a course.

        Algorithm:
        1. Fetch concepts (scoped to concept_ids > topic_id > whole course).
        2. For each concept, read BKT mastery (default if no row).
        3. Filter out fully mastered concepts (p_mastery >= MASTERY_THRESHOLD).
        4. Weighted random selection: weight = (1 - p_mastery). Lower mastery -> higher chance.
        5. Select up to MAX_CONCEPTS_PER_SESSION concepts.
        6. For each concept, fetch questions. Prioritize unseen, then least-recently-seen.
        7. Interleave and cap at SESSION_MAX_QUESTIONS.
        8. Return session payload (questions WITHOUT correct answers for the client).
        """
        # 1. Get concepts
        if concept_ids:
            all_concepts = await run_db_operation(
                lambda: _supabase.table("concepts")
                .select("id, name, explanation, complexity_level, topic_id")
                .in_("id", concept_ids)
                .execute()
            )
            all_concepts = getattr(all_concepts, "data", None) or []
        elif topic_id:
            all_concepts = await _select(
                "concepts", "id, name, explanation, complexity_level, topic_id",
                topic_id=topic_id,
            )
        else:
            all_concepts = await _get_concepts_for_course(course_id)

        if not all_concepts:
            return {"session_id": str(uuid.uuid4()), "questions": [], "concepts": [], "all_mastered": True}

        # 2. Fetch mastery rows for this course
        mastery_rows = await _select("bkt_mastery", "*", user_id=user_id, course_id=course_id)
        mastery_map: Dict[str, Dict[str, Any]] = {
            r["knowledge_component_id"]: r for r in mastery_rows
        }

        # 3. Build concept info with mastery + filter
        unmastered = []
        for c in all_concepts:
            row = mastery_map.get(c["id"])
            p = float(row["p_mastery"]) if row else DEFAULTS["p_mastery"]
            if p < MASTERY_THRESHOLD:
                unmastered.append({
                    "concept_id": c["id"],
                    "concept_name": c["name"],
                    "topic_id": c.get("topic_id"),
                    "p_mastery": p,
                    "n_attempts": int(row["n_attempts"]) if row else 0,
                })

        if not unmastered:
            return {"session_id": str(uuid.uuid4()), "questions": [], "concepts": [], "all_mastered": True}

        # 4. Weighted random selection
        weights = [max(0.01, 1.0 - c["p_mastery"]) for c in unmastered]
        total_weight = sum(weights)
        normalized = [w / total_weight for w in weights]

        n_pick = min(MAX_CONCEPTS_PER_SESSION, len(unmastered))
        selected_concepts: List[Dict[str, Any]] = []
        remaining_indices = list(range(len(unmastered)))

        for _ in range(n_pick):
            if not remaining_indices:
                break
            rem_weights = [normalized[i] for i in remaining_indices]
            rem_total = sum(rem_weights)
            if rem_total <= 0:
                break
            rem_probs = [w / rem_total for w in rem_weights]
            chosen_idx = random.choices(remaining_indices, weights=rem_probs, k=1)[0]
            selected_concepts.append(unmastered[chosen_idx])
            remaining_indices.remove(chosen_idx)

        selected_concept_ids = [c["concept_id"] for c in selected_concepts]

        # 5. Fetch questions for selected concepts
        all_questions = await _get_questions_for_concepts(selected_concept_ids)
        if not all_questions:
            return {
                "session_id": str(uuid.uuid4()),
                "questions": [],
                "concepts": selected_concepts,
                "all_mastered": False,
            }

        # 6. Filter recently-seen questions
        recent_ids = await _get_recent_question_ids(user_id, selected_concept_ids)
        unseen = [q for q in all_questions if q["id"] not in recent_ids]
        seen = [q for q in all_questions if q["id"] in recent_ids]

        def _group_by_concept(qs: list) -> Dict[str, list]:
            groups: Dict[str, list] = {}
            for q in qs:
                cid = q.get("concept_id", "")
                groups.setdefault(cid, []).append(q)
            return groups

        unseen_groups = _group_by_concept(unseen)
        seen_groups = _group_by_concept(seen)

        # Calculate questions per concept (proportional to 1-mastery, min 1)
        session_questions: list = []
        concept_weights = {c["concept_id"]: max(0.01, 1.0 - c["p_mastery"]) for c in selected_concepts}
        total_concept_weight = sum(concept_weights.values())
        questions_budget = SESSION_MAX_QUESTIONS

        for c in sorted(selected_concepts, key=lambda x: x["p_mastery"]):
            cid = c["concept_id"]
            proportion = concept_weights[cid] / total_concept_weight
            n_for_concept = max(1, round(proportion * questions_budget))

            pool = list(unseen_groups.get(cid, []))
            random.shuffle(pool)
            if len(pool) < n_for_concept:
                extra = list(seen_groups.get(cid, []))
                random.shuffle(extra)
                pool = pool + extra

            session_questions.extend(pool[:n_for_concept])

        # 7. Cap total and shuffle for interleaving
        session_questions = session_questions[:SESSION_MAX_QUESTIONS]
        random.shuffle(session_questions)

        # Ensure minimum session size
        if len(session_questions) < SESSION_MIN_QUESTIONS:
            all_available = unseen + seen
            existing_ids = {q["id"] for q in session_questions}
            extras = [q for q in all_available if q["id"] not in existing_ids]
            random.shuffle(extras)
            session_questions.extend(extras[: SESSION_MIN_QUESTIONS - len(session_questions)])

        # 8. Format for client (strip correct answer info)
        session_id = str(uuid.uuid4())
        formatted_questions = []
        for q in session_questions:
            options = q.get("question_options", [])
            if isinstance(options, list):
                options = sorted(options, key=lambda o: o.get("option_index", 0))

            formatted_questions.append({
                "id": q["id"],
                "question": q["question"],
                "hint": q.get("hint"),
                "difficulty_level": q.get("difficulty_level", "medium"),
                "concept_id": q.get("concept_id"),
                "concept_name": next(
                    (c["concept_name"] for c in selected_concepts if c["concept_id"] == q.get("concept_id")),
                    "Unknown",
                ),
                "options": [
                    {
                        "id": o.get("id", ""),
                        "text": o["option_text"],
                        "index": o["option_index"],
                        "is_correct": bool(o.get("is_correct", False)),
                        "explanation": o.get("explanation") or "",
                    }
                    for o in options
                ],
            })

        concept_summaries = [
            {
                "concept_id": c["concept_id"],
                "concept_name": c["concept_name"],
                "p_mastery": c["p_mastery"],
                "n_attempts": c["n_attempts"],
            }
            for c in selected_concepts
        ]

        return {
            "session_id": session_id,
            "questions": formatted_questions,
            "concepts": concept_summaries,
            "total_questions": len(formatted_questions),
            "all_mastered": False,
        }

    # -----------------------------------------------------------------------
    # Process a single answer (check + BKT update + record attempt)
    # -----------------------------------------------------------------------

    @staticmethod
    async def process_answer(
        user_id: str,
        question_id: str,
        course_id: str,
        selected_option_index: int,
        session_id: Optional[str] = None,
        time_spent_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process a student's answer:
        1. Look up whether the selected option is correct
        2. Record the attempt in user_question_attempts
        3. Run BKT update (correct -> score 100, wrong -> score 0)
        4. Return feedback: correctness, explanation, updated mastery, mastered flag
        """
        # 1. Get question + options
        qrow = await _select_single("questions", "id, question, concept_id, hint", id=question_id)
        if not qrow:
            raise ValueError(f"Question not found: {question_id}")

        concept_id = qrow.get("concept_id")

        options_resp = await run_db_operation(
            lambda: _supabase.table("question_options")
            .select("id, option_text, option_index, is_correct, explanation")
            .eq("question_id", question_id)
            .order("option_index")
            .execute()
        )
        options = getattr(options_resp, "data", None) or []
        if not options:
            raise ValueError(f"No options found for question {question_id}")

        selected_option = None
        correct_option = None
        for o in options:
            if o["option_index"] == selected_option_index:
                selected_option = o
            if o["is_correct"]:
                correct_option = o

        if not correct_option:
            raise ValueError(f"No correct option found for question {question_id}")

        is_correct = selected_option is not None and selected_option.get("is_correct", False)

        # 2. Record attempt (server-side audit trail)
        await _insert("user_question_attempts", {
            "user_id": user_id,
            "question_id": question_id,
            "concept_id": concept_id,
            "selected_option": selected_option_index,
            "is_correct": is_correct,
            "time_spent_ms": time_spent_ms,
            "session_id": session_id,
        })

        # 3. BKT update (scoped to course)
        claude_score = 100.0 if is_correct else 0.0
        bkt_result = await BKTService.update_mastery_for_response(
            user_id=user_id,
            question_id=question_id,
            course_id=course_id,
            claude_score=claude_score,
        )

        p_mastery_after = DEFAULTS["p_mastery"]
        p_mastery_before = DEFAULTS["p_mastery"]
        if bkt_result.get("updated"):
            first = bkt_result["updated"][0]
            p_mastery_after = first.get("p_mastery_after", DEFAULTS["p_mastery"])
            p_mastery_before = first.get("p_mastery_before", DEFAULTS["p_mastery"])

        is_newly_mastered = p_mastery_before < MASTERY_THRESHOLD <= p_mastery_after

        return {
            "question_id": question_id,
            "concept_id": concept_id,
            "is_correct": is_correct,
            "correct_option_index": correct_option["option_index"],
            "correct_option_text": correct_option["option_text"],
            "explanation": correct_option.get("explanation", ""),
            "selected_option_index": selected_option_index,
            "p_mastery_before": p_mastery_before,
            "p_mastery_after": p_mastery_after,
            "is_newly_mastered": is_newly_mastered,
            "mastery_threshold": MASTERY_THRESHOLD,
        }

    # -----------------------------------------------------------------------
    # Course progress tree (for the overview page)
    # -----------------------------------------------------------------------

    @staticmethod
    async def get_course_progress(user_id: str, course_id: str) -> Dict[str, Any]:
        """
        Full mastery progress tree: course -> topics -> concepts.
        Each concept has its BKT p_mastery. Topics and course have aggregate progress.
        """
        course = await _select_single("courses", "id, title", id=course_id)
        if not course:
            raise ValueError(f"Course not found: {course_id}")

        topics = await _get_topics_for_course(course_id)
        if not topics:
            return {
                "course_id": course_id,
                "course_title": course.get("title", "Untitled"),
                "topics": [],
                "total_concepts": 0,
                "mastered_concepts": 0,
                "overall_progress": 0,
                "mastery_threshold": MASTERY_THRESHOLD,
            }

        all_concepts = await _get_concepts_for_course(course_id)
        concept_ids = [c["id"] for c in all_concepts]

        mastery_rows = await _select("bkt_mastery", "*", user_id=user_id, course_id=course_id)
        mastery_map = {r["knowledge_component_id"]: r for r in mastery_rows}

        # Question counts per concept
        question_counts: Dict[str, int] = {}
        if concept_ids:
            qc_resp = await run_db_operation(
                lambda: _supabase.table("questions")
                .select("concept_id")
                .in_("concept_id", concept_ids)
                .execute()
            )
            for row in (getattr(qc_resp, "data", None) or []):
                cid = row.get("concept_id")
                if cid:
                    question_counts[cid] = question_counts.get(cid, 0) + 1

        concepts_by_topic: Dict[str, List[Dict]] = {}
        for c in all_concepts:
            concepts_by_topic.setdefault(c["topic_id"], []).append(c)

        topic_progress_list = []
        total_concepts = 0
        total_mastered = 0

        for topic in topics:
            tid = topic["id"]
            topic_concepts = concepts_by_topic.get(tid, [])
            concept_list = []

            for c in topic_concepts:
                cid = c["id"]
                row = mastery_map.get(cid)
                p = float(row["p_mastery"]) if row else DEFAULTS["p_mastery"]
                n_att = int(row["n_attempts"]) if row else 0
                n_cor = int(row["n_correct"]) if row else 0
                is_mastered = p >= MASTERY_THRESHOLD

                if is_mastered:
                    status = "mastered"
                elif n_att > 0:
                    status = "in_progress"
                else:
                    status = "not_started"

                concept_list.append({
                    "concept_id": cid,
                    "concept_name": c["name"],
                    "explanation": c.get("explanation", ""),
                    "p_mastery": round(p, 4),
                    "n_attempts": n_att,
                    "n_correct": n_cor,
                    "status": status,
                    "is_mastered": is_mastered,
                    "question_count": question_counts.get(cid, 0),
                })
                total_concepts += 1
                if is_mastered:
                    total_mastered += 1

            topic_mastered = sum(1 for c in concept_list if c["is_mastered"])
            topic_total = len(concept_list)
            topic_progress = round((topic_mastered / topic_total) * 100) if topic_total > 0 else 0

            if topic_mastered == topic_total and topic_total > 0:
                topic_status = "mastered"
            elif any(c["status"] != "not_started" for c in concept_list):
                topic_status = "in_progress"
            else:
                topic_status = "not_started"

            topic_progress_list.append({
                "topic_id": tid,
                "topic_name": topic["name"],
                "status": topic_status,
                "concepts": concept_list,
                "total_concepts": topic_total,
                "mastered_concepts": topic_mastered,
                "overall_progress": topic_progress,
            })

        overall = round((total_mastered / total_concepts) * 100) if total_concepts > 0 else 0

        return {
            "course_id": course_id,
            "course_title": course.get("title", "Untitled"),
            "topics": topic_progress_list,
            "total_concepts": total_concepts,
            "mastered_concepts": total_mastered,
            "overall_progress": overall,
            "mastery_threshold": MASTERY_THRESHOLD,
        }

    # -----------------------------------------------------------------------
    # Legacy-style helpers (now course-scoped)
    # -----------------------------------------------------------------------

    @staticmethod
    async def get_mastery_for_course(user_id: str, course_id: str) -> Dict[str, Any]:
        concepts = await _get_concepts_for_course(course_id)
        skills: List[Dict[str, Any]] = []
        p_list: List[float] = []

        mastery_rows = await _select("bkt_mastery", "*", user_id=user_id, course_id=course_id)
        mastery_map = {r["knowledge_component_id"]: r for r in mastery_rows}

        for c in concepts:
            kc_id = str(c["id"])
            row = mastery_map.get(kc_id)
            p = float(row["p_mastery"]) if row else DEFAULTS["p_mastery"]
            attempts = int(row["n_attempts"]) if row else 0
            p_list.append(p)
            skills.append({
                "skill_name": c.get("name") or kc_id,
                "mastery": p,
                "attempts": attempts,
            })

        total_attempts = sum(s["attempts"] for s in skills)

        # Fetch the course's target_grade for the pass probability calculation
        target_grade = 1.0
        try:
            course_resp = await run_db_operation(
                lambda: _supabase.table("courses").select("target_grade").eq("id", course_id).maybe_single().execute()
            )
            course_data = getattr(course_resp, "data", None)
            if course_data and course_data.get("target_grade") is not None:
                target_grade = float(course_data["target_grade"])
        except Exception as e:
            logger.warning(f"Failed to fetch target_grade for course {course_id}: {e}")

        return {
            "pass_probability": aggregate_pass_probability(p_list, target=target_grade) if total_attempts > 0 else None,
            "target_grade": target_grade,
            "skills": skills,
        }

    @staticmethod
    async def get_weak_skills_for_course(user_id: str, course_id: str, limit: int = 5) -> Dict[str, Any]:
        mastery = await BKTService.get_mastery_for_course(user_id, course_id)
        weakest = sorted(mastery["skills"], key=lambda s: float(s["mastery"]))[:limit]
        return {"skills": weakest}
