"""
On-demand quiz service.

Generates a fresh named quiz (course_quizzes record) with N BKT-guided questions.
The quiz is saved to DB; the user then starts an attempt via quiz_attempts_service.
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic
from app.core.config import get_settings
from app.core.supabase import get_supabase
from app.core.async_db import run_db_operation
from app.services.question_generator import QuestionGenerator

logger = logging.getLogger(__name__)

_supabase = get_supabase()
_question_generator = QuestionGenerator()
_settings = get_settings()

MAX_CONCURRENT_GENERATIONS = 5
DEFAULT_QUIZ_SIZE = 10


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _get_concepts_for_course(course_id: str) -> List[Dict[str, Any]]:
    docs_resp = await run_db_operation(
        lambda: _supabase.table("documents")
        .select("id")
        .eq("course_id", course_id)
        .eq("status", "completed")
        .execute()
    )
    docs = getattr(docs_resp, "data", None) or []
    if not docs:
        return []

    doc_ids = [d["id"] for d in docs]
    topics_resp = await run_db_operation(
        lambda: _supabase.table("topics")
        .select("id")
        .in_("document_id", doc_ids)
        .execute()
    )
    topics = getattr(topics_resp, "data", None) or []
    if not topics:
        return []

    topic_ids = [t["id"] for t in topics]
    concepts_resp = await run_db_operation(
        lambda: _supabase.table("concepts")
        .select("id, name, explanation, source_text, topic_id")
        .in_("topic_id", topic_ids)
        .execute()
    )
    return getattr(concepts_resp, "data", None) or []


async def _get_mastery_map(user_id: str, course_id: str) -> Dict[str, float]:
    resp = await run_db_operation(
        lambda: _supabase.table("bkt_mastery")
        .select("knowledge_component_id, p_mastery")
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .execute()
    )
    rows = getattr(resp, "data", None) or []
    return {r["knowledge_component_id"]: r["p_mastery"] for r in rows}


# ---------------------------------------------------------------------------
# Concept selection
# ---------------------------------------------------------------------------

def _select_concepts(
    concepts: List[Dict[str, Any]],
    mastery_map: Dict[str, float],
    quiz_size: int,
) -> List[Dict[str, Any]]:
    if not concepts:
        return []
    n = min(quiz_size, len(concepts))
    weights = [max(0.05, 1.0 - mastery_map.get(c["id"], 0.2)) for c in concepts]
    selected = random.choices(concepts, weights=weights, k=n * 3)
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for c in selected:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)
        if len(unique) == n:
            break
    if len(unique) < n:
        remaining = [c for c in concepts if c["id"] not in seen]
        unique.extend(remaining[: n - len(unique)])
    return unique


# ---------------------------------------------------------------------------
# AI quiz naming
# ---------------------------------------------------------------------------

async def _generate_quiz_name(concept_names: List[str]) -> str:
    try:
        client = AsyncAnthropic(api_key=_settings.ANTHROPIC_API_KEY)
        concepts_str = ", ".join(concept_names[:8])
        prompt = (
            f"Generate a short, engaging quiz title (3–7 words) for a quiz covering: {concepts_str}. "
            "The title should feel like a book chapter or course module — specific but evocative. "
            "Examples: 'The Chemistry of Cellular Energy', 'Market Forces in Practice', "
            "'Into the Nervous System'. Return only the title, nothing else."
        )
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=40,
            messages=[{"role": "user", "content": prompt}],
        )
        name = response.content[0].text.strip().strip('"').strip("'")
        return name if name else "Quiz"
    except Exception as e:
        logger.warning(f"AI name generation failed: {e}")
        return "Quiz"


# ---------------------------------------------------------------------------
# Question generation + saving
# ---------------------------------------------------------------------------

async def _generate_and_save_question(
    concept: Dict[str, Any],
    quiz_id: str,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """Generate 1 question for a concept, save it to DB. Returns question_id or None."""
    async with semaphore:
        try:
            questions = await _question_generator.generate_questions_for_concept(
                concept_id=concept["id"],
                concept_name=concept["name"],
                concept_explanation=concept.get("explanation") or "",
                source_text=concept.get("source_text") or "",
                num_questions=1,
            )
            if not questions:
                logger.warning(f"No question generated for concept: {concept['name']}")
                return None

            q = questions[0]

            options_list = list(q.options)
            random.shuffle(options_list)
            correct_idx = next(i for i, o in enumerate(options_list) if o.is_correct)

            q_resp = await run_db_operation(
                lambda _q=q, _ci=correct_idx: _supabase.table("questions").insert({
                    "quiz_id": None,
                    "course_quiz_id": quiz_id,
                    "question": _q.question,
                    "options": [],
                    "correct_answer": _ci,
                    "explanation": "",
                    "order_index": 0,
                    "concept_id": _q.concept_id,
                    "hint": _q.hint,
                    "difficulty_level": _q.difficulty_level,
                }).execute()
            )
            q_data = getattr(q_resp, "data", None)
            if not q_data:
                logger.error("Failed to insert question row")
                return None

            question_id = q_data[0]["id"]

            options_data = [
                {
                    "question_id": question_id,
                    "option_text": o.option_text,
                    "option_index": idx,
                    "is_correct": o.is_correct,
                    "explanation": o.explanation,
                }
                for idx, o in enumerate(options_list)
            ]
            await run_db_operation(
                lambda od=options_data: _supabase.table("question_options").insert(od).execute()
            )
            return question_id

        except Exception as e:
            logger.error(f"Error generating/saving question for '{concept['name']}': {e}")
            return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def generate_quiz(
    user_id: str,
    course_id: str,
    quiz_size: int = DEFAULT_QUIZ_SIZE,
) -> Dict[str, Any]:
    """
    Generate a fresh named quiz for a user+course.

    Returns {quiz_id, name, total_questions, course_id}.
    The caller is responsible for starting an attempt via quiz_attempts_service.
    """
    logger.info(f"Generating quiz: user={user_id} course={course_id} size={quiz_size}")

    concepts = await _get_concepts_for_course(course_id)
    if not concepts:
        return {
            "quiz_id": None,
            "name": None,
            "total_questions": 0,
            "course_id": course_id,
            "error": "No concepts found. Upload and process documents first.",
        }

    mastery_map = await _get_mastery_map(user_id, course_id)
    selected_concepts = _select_concepts(concepts, mastery_map, quiz_size)

    quiz_id = str(uuid.uuid4())

    # Create the course_quizzes row FIRST so questions can reference it via FK.
    # Name and total_questions are updated after generation completes.
    try:
        await run_db_operation(
            lambda: _supabase.table("course_quizzes").insert({
                "id": quiz_id,
                "course_id": course_id,
                "user_id": user_id,
                "name": "Generating…",
                "total_questions": 0,
                "question_order": [],
            }).execute()
        )
    except Exception as e:
        logger.error(f"Failed to create course_quizzes placeholder: {e}")
        return {
            "quiz_id": None,
            "name": None,
            "total_questions": 0,
            "course_id": course_id,
            "error": "Failed to save quiz. Please try again.",
        }

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
    tasks = [_generate_and_save_question(c, quiz_id, semaphore) for c in selected_concepts]
    results = await asyncio.gather(*tasks)

    question_ids = [qid for qid in results if qid is not None]
    if not question_ids:
        logger.error("Quiz generation produced no questions")
        # Clean up the placeholder row
        try:
            await run_db_operation(
                lambda: _supabase.table("course_quizzes").delete().eq("id", quiz_id).execute()
            )
        except Exception:
            pass
        return {
            "quiz_id": None,
            "name": None,
            "total_questions": 0,
            "course_id": course_id,
            "error": "Could not generate questions. Please try again.",
        }

    random.shuffle(question_ids)

    concept_names = [c["name"] for c in selected_concepts]
    quiz_name = await _generate_quiz_name(concept_names)

    # Update the row with final name, count, and question order
    try:
        await run_db_operation(
            lambda: _supabase.table("course_quizzes").update({
                "name": quiz_name,
                "total_questions": len(question_ids),
                "question_order": question_ids,
            }).eq("id", quiz_id).execute()
        )
    except Exception as e:
        logger.error(f"Failed to update course_quizzes record: {e}")

    logger.info(f"Quiz {quiz_id} '{quiz_name}' created with {len(question_ids)} questions")

    return {
        "quiz_id": quiz_id,
        "name": quiz_name,
        "total_questions": len(question_ids),
        "course_id": course_id,
    }
