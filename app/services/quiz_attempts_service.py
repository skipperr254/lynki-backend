"""
Quiz attempts service.

Handles starting, answering, completing, and resuming attempts on course_quizzes.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.supabase import get_supabase
from app.core.async_db import run_db_operation
from app.services.bkt.service import BKTService, DEFAULTS, MASTERY_THRESHOLD

logger = logging.getLogger(__name__)
_supabase = get_supabase()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _fetch_quiz_questions_ordered(quiz_id: str) -> List[Dict[str, Any]]:
    """Fetch questions for a quiz in the original shuffled order."""
    quiz_resp = await run_db_operation(
        lambda: _supabase.table("course_quizzes")
        .select("question_order")
        .eq("id", quiz_id)
        .maybe_single()
        .execute()
    )
    question_order: List[str] = (getattr(quiz_resp, "data", None) or {}).get("question_order", [])

    if not question_order:
        return []

    q_resp = await run_db_operation(
        lambda: _supabase.table("questions")
        .select("id, question, hint, difficulty_level, concept_id, question_options(id, option_text, option_index, is_correct, explanation)")
        .in_("id", question_order)
        .execute()
    )
    raw = getattr(q_resp, "data", None) or []
    q_map = {q["id"]: q for q in raw}
    return [q_map[qid] for qid in question_order if qid in q_map]


async def _get_concept_names(raw_questions: List[Dict]) -> Dict[str, str]:
    concept_ids = {q.get("concept_id") for q in raw_questions if q.get("concept_id")}
    if not concept_ids:
        return {}
    resp = await run_db_operation(
        lambda: _supabase.table("concepts")
        .select("id, name")
        .in_("id", list(concept_ids))
        .execute()
    )
    return {c["id"]: c.get("name", "Unknown") for c in (getattr(resp, "data", None) or [])}


def _format_questions(
    raw_questions: List[Dict], concept_names: Dict[str, str]
) -> List[Dict[str, Any]]:
    formatted = []
    for q in raw_questions:
        options = sorted(
            q.get("question_options", []),
            key=lambda o: o.get("option_index", 0),
        )
        formatted.append({
            "id": q["id"],
            "question": q["question"],
            "concept_id": q.get("concept_id"),
            "concept_name": concept_names.get(q.get("concept_id", ""), "Unknown"),
            "difficulty_level": q.get("difficulty_level", "medium"),
            "hint": q.get("hint"),
            "options": [
                {
                    "id": o.get("id", ""),
                    "text": o["option_text"],
                    "index": o["option_index"],
                    "is_correct": o.get("is_correct", False),
                    "explanation": o.get("explanation", ""),
                }
                for o in options
            ],
        })
    return formatted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def start_quiz_attempt(user_id: str, quiz_id: str, course_id: str) -> Dict[str, Any]:
    """
    Start a new attempt on a quiz. Creates a quiz_attempts row.
    Returns TestData-shaped payload (test_id = attempt_id).
    """
    quiz_resp = await run_db_operation(
        lambda: _supabase.table("course_quizzes")
        .select("id, name, status, error_message")
        .eq("id", quiz_id)
        .maybe_single()
        .execute()
    )
    quiz = getattr(quiz_resp, "data", None)
    if not quiz:
        raise ValueError(f"Quiz not found: {quiz_id}")

    # Defensive backstop — the frontend polls course_quizzes.status and only
    # calls start once it's 'completed', but this guards against stale links,
    # a second tab, or a race, instead of silently creating a 0-question
    # attempt (the previous behavior when question_order wasn't populated yet).
    status = quiz.get("status")
    if status == "generating":
        raise ValueError("This quiz is still generating. Please wait a moment and try again.")
    if status == "failed":
        raise ValueError(quiz.get("error_message") or "Quiz generation failed. Please generate a new quiz.")

    course_resp = await run_db_operation(
        lambda: _supabase.table("courses")
        .select("title")
        .eq("id", course_id)
        .maybe_single()
        .execute()
    )
    course_name = (getattr(course_resp, "data", None) or {}).get("title", quiz["name"])

    attempt_id = str(uuid.uuid4())
    await run_db_operation(
        lambda: _supabase.table("quiz_attempts").insert({
            "id": attempt_id,
            "quiz_id": quiz_id,
            "user_id": user_id,
            "course_id": course_id,
            "status": "in_progress",
            "answered_count": 0,
            "correct_count": 0,
        }).execute()
    )

    raw_questions = await _fetch_quiz_questions_ordered(quiz_id)
    concept_names = await _get_concept_names(raw_questions)
    formatted = _format_questions(raw_questions, concept_names)

    return {
        "test_id": attempt_id,
        "quiz_id": quiz_id,
        "course_id": course_id,
        "course_name": course_name,
        "questions": formatted,
        "total_questions": len(formatted),
    }


async def resume_quiz_attempt(user_id: str, quiz_attempt_id: str) -> Dict[str, Any]:
    """
    Resume an in-progress quiz attempt.
    Returns all questions + answered_count so TestPage skips to the right index.
    """
    attempt_resp = await run_db_operation(
        lambda: _supabase.table("quiz_attempts")
        .select("id, quiz_id, course_id, status, answered_count, correct_count")
        .eq("id", quiz_attempt_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    attempt = getattr(attempt_resp, "data", None)
    if not attempt:
        raise ValueError(f"Attempt not found: {quiz_attempt_id}")
    if attempt["status"] == "completed":
        raise ValueError("This attempt has already been completed")

    quiz_id = attempt["quiz_id"]
    course_id = attempt["course_id"]

    quiz_resp = await run_db_operation(
        lambda: _supabase.table("course_quizzes")
        .select("name")
        .eq("id", quiz_id)
        .maybe_single()
        .execute()
    )
    quiz_name = (getattr(quiz_resp, "data", None) or {}).get("name", "Quiz")

    course_resp = await run_db_operation(
        lambda: _supabase.table("courses")
        .select("title")
        .eq("id", course_id)
        .maybe_single()
        .execute()
    )
    course_name = (getattr(course_resp, "data", None) or {}).get("title", quiz_name)

    raw_questions = await _fetch_quiz_questions_ordered(quiz_id)
    concept_names = await _get_concept_names(raw_questions)
    formatted = _format_questions(raw_questions, concept_names)

    return {
        "test_id": quiz_attempt_id,
        "quiz_id": quiz_id,
        "course_id": course_id,
        "course_name": course_name,
        "questions": formatted,
        "total_questions": len(formatted),
        "answered_count": attempt.get("answered_count", 0),
        "correct_count": attempt.get("correct_count", 0),
    }


async def process_quiz_answer(
    user_id: str,
    course_id: str,
    quiz_attempt_id: str,
    question_id: str,
    selected_option_index: int,
) -> Dict[str, Any]:
    """
    Process a student's answer within a quiz attempt.
    Writes to question_attempts, updates quiz_attempts counts, runs BKT update.
    Returns AnswerFeedback-shaped dict.
    """
    # Look up options
    options_resp = await run_db_operation(
        lambda: _supabase.table("question_options")
        .select("id, option_text, option_index, is_correct, explanation")
        .eq("question_id", question_id)
        .order("option_index")
        .execute()
    )
    options = getattr(options_resp, "data", None) or []
    if not options:
        raise ValueError(f"No options for question {question_id}")

    selected_option = None
    correct_option = None
    for o in options:
        if o["option_index"] == selected_option_index:
            selected_option = o
        if o["is_correct"]:
            correct_option = o

    if not correct_option:
        raise ValueError(f"No correct option for question {question_id}")

    is_correct = selected_option is not None and selected_option.get("is_correct", False)

    # concept_id for BKT
    qrow_resp = await run_db_operation(
        lambda: _supabase.table("questions")
        .select("concept_id")
        .eq("id", question_id)
        .maybe_single()
        .execute()
    )
    concept_id = (getattr(qrow_resp, "data", None) or {}).get("concept_id")

    # Record question attempt
    await run_db_operation(
        lambda: _supabase.table("question_attempts").insert({
            "quiz_attempt_id": quiz_attempt_id,
            "question_id": question_id,
            "user_id": user_id,
            "course_id": course_id,
            "selected_option_index": selected_option_index,
            "is_correct": is_correct,
        }).execute()
    )

    # Update quiz attempt counts
    attempt_resp = await run_db_operation(
        lambda: _supabase.table("quiz_attempts")
        .select("answered_count, correct_count, quiz_id")
        .eq("id", quiz_attempt_id)
        .maybe_single()
        .execute()
    )
    attempt = getattr(attempt_resp, "data", None) or {}
    new_answered = (attempt.get("answered_count") or 0) + 1
    new_correct = (attempt.get("correct_count") or 0) + (1 if is_correct else 0)

    # Check if this was the last question
    quiz_resp = await run_db_operation(
        lambda: _supabase.table("course_quizzes")
        .select("total_questions")
        .eq("id", attempt.get("quiz_id", ""))
        .maybe_single()
        .execute()
    )
    total_q = (getattr(quiz_resp, "data", None) or {}).get("total_questions", 0)

    update_data: Dict[str, Any] = {
        "answered_count": new_answered,
        "correct_count": new_correct,
    }
    if total_q > 0 and new_answered >= total_q:
        update_data["status"] = "completed"
        update_data["completed_at"] = datetime.now(timezone.utc).isoformat()
        try:
            mastery = await BKTService.get_mastery_for_course(user_id, course_id)
            update_data["pass_chance"] = mastery["pass_probability"]
        except Exception:
            pass

    await run_db_operation(
        lambda: _supabase.table("quiz_attempts")
        .update(update_data)
        .eq("id", quiz_attempt_id)
        .execute()
    )

    # BKT update
    p_mastery_before = DEFAULTS["p_mastery"]
    p_mastery_after = DEFAULTS["p_mastery"]
    is_newly_mastered = False
    try:
        bkt_result = await BKTService.update_mastery_for_response(
            user_id=user_id,
            question_id=question_id,
            course_id=course_id,
            claude_score=100.0 if is_correct else 0.0,
        )
        if bkt_result.get("updated"):
            first = bkt_result["updated"][0]
            p_mastery_before = first.get("p_mastery_before", DEFAULTS["p_mastery"])
            p_mastery_after = first.get("p_mastery_after", DEFAULTS["p_mastery"])
            is_newly_mastered = p_mastery_before < MASTERY_THRESHOLD <= p_mastery_after
    except Exception as e:
        logger.warning(f"BKT update failed for question {question_id}: {e}")

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


async def complete_quiz_attempt(
    user_id: str, course_id: str, quiz_attempt_id: str
) -> Dict[str, Any]:
    """Explicitly mark a quiz attempt as completed and record pass_chance."""
    try:
        mastery = await BKTService.get_mastery_for_course(user_id, course_id)
        pass_probability: Optional[float] = mastery["pass_probability"]
    except Exception:
        pass_probability = None

    await run_db_operation(
        lambda: _supabase.table("quiz_attempts")
        .update({
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "pass_chance": pass_probability,
        })
        .eq("id", quiz_attempt_id)
        .execute()
    )

    return {
        "attempt_id": quiz_attempt_id,
        "status": "completed",
        "pass_chance": pass_probability,
    }
