"""
Test Service — generates full-coverage quizzes and processes answers.

Core flow:
  1. Generate a quiz: one question per concept across the entire course.
     Avoids recently-answered questions where possible.
     Session is persisted in test_sessions table.
  2. Process each answer: check correctness, run BKT update, record attempt,
     update the test session's answered_count / correct_count.
  3. After all answers submitted, frontend fetches pass probability.
"""

from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import json

from app.core.supabase import get_supabase
from app.core.async_db import db_select, db_insert, db_update, run_db_operation
from app.services.bkt.service import (
    BKTService,
    aggregate_pass_probability,
    _get_concepts_for_course,
    _get_questions_for_concepts,
    _select,
    DEFAULTS,
)

logger = logging.getLogger(__name__)

_supabase = get_supabase()

# How far back to look when avoiding repeated questions (hours)
RECENTLY_SEEN_HOURS = 24


async def _get_user_recent_questions(
    user_id: str, concept_ids: List[str], hours: int = RECENTLY_SEEN_HOURS
) -> Dict[str, List[str]]:
    """
    Get question IDs the user has answered recently, grouped by concept_id.
    Returns {concept_id: [question_id, ...]}.
    """
    if not concept_ids:
        return {}
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    resp = await run_db_operation(
        lambda: _supabase.table("user_question_attempts")
        .select("question_id, concept_id")
        .eq("user_id", user_id)
        .in_("concept_id", concept_ids)
        .gte("created_at", cutoff)
        .execute()
    )
    data = getattr(resp, "data", None) or []
    result: Dict[str, List[str]] = {}
    for row in data:
        cid = row.get("concept_id")
        qid = row.get("question_id")
        if cid and qid:
            result.setdefault(cid, []).append(qid)
    return result


async def generate_test(user_id: str, course_id: str) -> Dict[str, Any]:
    """
    Generate a full-coverage test for a course.

    Algorithm:
    1. Fetch all concepts for the course
    2. For each concept, fetch all available questions
    3. Pick one question per concept, preferring unseen ones
    4. Shuffle the final question list
    5. Return as a test payload
    """
    # 0. Get course name
    course_resp = await run_db_operation(
        lambda: _supabase.table("courses").select("title").eq("id", course_id).maybe_single().execute()
    )
    course_name = (getattr(course_resp, "data", None) or {}).get("title", "Quiz")

    # 1. Get all concepts
    concepts = await _get_concepts_for_course(course_id)
    if not concepts:
        return {
            "test_id": str(uuid.uuid4()),
            "course_id": course_id,
            "course_name": course_name,
            "questions": [],
            "total_questions": 0,
            "message": "No concepts found for this course. Upload and process documents first.",
        }

    concept_ids = [c["id"] for c in concepts]

    # 2. Fetch all questions for all concepts
    all_questions = await _get_questions_for_concepts(concept_ids)
    if not all_questions:
        return {
            "test_id": str(uuid.uuid4()),
            "course_id": course_id,
            "course_name": course_name,
            "questions": [],
            "total_questions": 0,
            "message": "No questions available yet. Documents may still be processing.",
        }

    # 3. Group questions by concept
    questions_by_concept: Dict[str, List[Dict]] = {}
    for q in all_questions:
        cid = q.get("concept_id", "")
        if cid:
            questions_by_concept.setdefault(cid, []).append(q)

    # 4. Get recently-seen questions to avoid repeats
    recent_by_concept = await _get_user_recent_questions(user_id, concept_ids)

    # 5. Pick one question per concept
    selected_questions: List[Dict] = []
    concept_name_map = {c["id"]: c.get("name", "Unknown") for c in concepts}

    for concept_id in concept_ids:
        pool = questions_by_concept.get(concept_id, [])
        if not pool:
            continue

        recent_ids = set(recent_by_concept.get(concept_id, []))

        # Prefer unseen questions
        unseen = [q for q in pool if q["id"] not in recent_ids]
        if unseen:
            chosen = random.choice(unseen)
        else:
            # All seen — pick random from full pool
            chosen = random.choice(pool)

        selected_questions.append(chosen)

    # 6. Shuffle for variety
    random.shuffle(selected_questions)

    # 7. Format for client
    test_id = str(uuid.uuid4())
    formatted_questions = []
    question_id_list = []
    for q in selected_questions:
        question_id_list.append(q["id"])
        options = q.get("question_options", [])
        if isinstance(options, list):
            options = sorted(options, key=lambda o: o.get("option_index", 0))

        formatted_questions.append({
            "id": q["id"],
            "question": q["question"],
            "concept_id": q.get("concept_id"),
            "concept_name": concept_name_map.get(q.get("concept_id", ""), "Unknown"),
            "difficulty_level": q.get("difficulty_level", "medium"),
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

    # 8. Persist the session to DB
    try:
        await db_insert(_supabase, "test_sessions", {
            "id": test_id,
            "user_id": user_id,
            "course_id": course_id,
            "status": "in_progress",
            "total_questions": len(formatted_questions),
            "correct_count": 0,
            "answered_count": 0,
            "question_ids": json.dumps(question_id_list),
        })
    except Exception as e:
        logger.warning(f"Failed to persist test session: {e}")

    return {
        "test_id": test_id,
        "course_id": course_id,
        "course_name": course_name,
        "questions": formatted_questions,
        "total_questions": len(formatted_questions),
    }


async def process_test_answer(
    user_id: str,
    course_id: str,
    question_id: str,
    selected_option_index: int,
    test_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a single test answer.
    Delegates to BKTService.process_answer for correctness check + BKT update.
    Also updates the test_session counts if test_id is provided.
    """
    result = await BKTService.process_answer(
        user_id=user_id,
        question_id=question_id,
        course_id=course_id,
        selected_option_index=selected_option_index,
        session_id=test_id,
    )

    # Update session progress
    if test_id:
        try:
            # Increment answered_count, and correct_count if correct
            session_rows = await run_db_operation(
                lambda: _supabase.table("test_sessions")
                .select("answered_count, correct_count, total_questions")
                .eq("id", test_id)
                .maybe_single()
                .execute()
            )
            session = getattr(session_rows, "data", None)
            if session:
                new_answered = (session.get("answered_count") or 0) + 1
                new_correct = (session.get("correct_count") or 0) + (1 if result.get("is_correct") else 0)
                update_data: Dict[str, Any] = {
                    "answered_count": new_answered,
                    "correct_count": new_correct,
                }
                # Auto-complete if all questions answered
                if new_answered >= (session.get("total_questions") or 0):
                    update_data["status"] = "completed"
                    update_data["completed_at"] = datetime.now(timezone.utc).isoformat()
                    # Compute and store pass chance
                    try:
                        mastery = await BKTService.get_mastery_for_course(user_id, course_id)
                        update_data["pass_chance"] = mastery["pass_probability"]
                    except Exception:
                        pass
                await db_update(_supabase, "test_sessions", update_data, id=test_id)
        except Exception as e:
            logger.warning(f"Failed to update test session {test_id}: {e}")

    return result


async def complete_test_session(
    user_id: str,
    course_id: str,
    test_id: str,
) -> Dict[str, Any]:
    """
    Explicitly complete a test session and record the final pass chance.
    Called when the user finishes the quiz.
    """
    try:
        mastery = await BKTService.get_mastery_for_course(user_id, course_id)
        pass_probability = mastery["pass_probability"]
    except Exception:
        pass_probability = None

    try:
        await db_update(
            _supabase,
            "test_sessions",
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "pass_chance": pass_probability,
            },
            id=test_id,
        )
    except Exception as e:
        logger.warning(f"Failed to complete test session {test_id}: {e}")

    return {
        "test_id": test_id,
        "status": "completed",
        "pass_chance": pass_probability,
    }


async def get_test_history(
    user_id: str,
    course_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Get the quiz history for a user in a course.
    Returns test sessions ordered by most recent first.
    """
    resp = await run_db_operation(
        lambda: _supabase.table("test_sessions")
        .select("id, status, total_questions, correct_count, answered_count, pass_chance, created_at, completed_at")
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return getattr(resp, "data", None) or []


async def resume_test(user_id: str, session_id: str) -> Dict[str, Any]:
    """
    Resume an in-progress test session.

    1. Fetch session metadata from test_sessions
    2. Reconstruct the full question list from stored question_ids
    3. Fetch which questions the user already answered in this session
    4. Return the same TestData shape plus resume state (answered_count, correct_count)
    """
    # 1. Get session
    session_resp = await run_db_operation(
        lambda: _supabase.table("test_sessions")
        .select("*")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    session = getattr(session_resp, "data", None)
    if not session:
        raise ValueError(f"Session not found: {session_id}")

    if session.get("status") == "completed":
        raise ValueError("This quiz has already been completed")

    course_id = session["course_id"]

    # Fetch course name
    course_resp = await run_db_operation(
        lambda: _supabase.table("courses").select("title").eq("id", course_id).maybe_single().execute()
    )
    course_name = (getattr(course_resp, "data", None) or {}).get("title", "Quiz")

    question_ids = session.get("question_ids", [])
    if isinstance(question_ids, str):
        question_ids = json.loads(question_ids)

    if not question_ids:
        return {
            "test_id": session_id,
            "course_id": course_id,
            "course_name": course_name,
            "questions": [],
            "total_questions": 0,
            "answered_count": 0,
            "correct_count": 0,
            "message": "No questions in this session.",
        }

    # 2. Fetch all questions by IDs
    questions_resp = await run_db_operation(
        lambda: _supabase.table("questions")
        .select("id, question, hint, difficulty_level, concept_id, question_options(id, option_text, option_index, is_correct, explanation)")
        .in_("id", question_ids)
        .execute()
    )
    raw_questions = getattr(questions_resp, "data", None) or []

    # Build a lookup so we can restore the original order from question_ids
    q_map = {q["id"]: q for q in raw_questions}

    # 3. Get concept names
    concept_ids_set = {q.get("concept_id") for q in raw_questions if q.get("concept_id")}
    concept_names: Dict[str, str] = {}
    if concept_ids_set:
        concepts_resp = await run_db_operation(
            lambda: _supabase.table("concepts")
            .select("id, name")
            .in_("id", list(concept_ids_set))
            .execute()
        )
        for c in getattr(concepts_resp, "data", None) or []:
            concept_names[c["id"]] = c.get("name", "Unknown")

    # 4. Format questions in original order
    formatted_questions = []
    for qid in question_ids:
        q = q_map.get(qid)
        if not q:
            continue
        options = q.get("question_options", [])
        if isinstance(options, list):
            options = sorted(options, key=lambda o: o.get("option_index", 0))

        formatted_questions.append({
            "id": q["id"],
            "question": q["question"],
            "concept_id": q.get("concept_id"),
            "concept_name": concept_names.get(q.get("concept_id", ""), "Unknown"),
            "difficulty_level": q.get("difficulty_level", "medium"),
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

    return {
        "test_id": session_id,
        "course_id": course_id,
        "course_name": course_name,
        "questions": formatted_questions,
        "total_questions": len(formatted_questions),
        "answered_count": session.get("answered_count", 0),
        "correct_count": session.get("correct_count", 0),
    }


async def get_pass_chance(user_id: str, course_id: str) -> Dict[str, Any]:
    """
    Get the current pass probability for a user in a course.
    Uses BKT mastery data to compute aggregate pass probability.
    """
    result = await BKTService.get_mastery_for_course(user_id, course_id)
    return {
        "course_id": course_id,
        "pass_probability": result["pass_probability"],
        "target_grade": result.get("target_grade", 1.0),
        "total_skills": len(result["skills"]),
        "skills": result["skills"],
    }
