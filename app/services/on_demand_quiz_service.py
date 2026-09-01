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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic
from app.core.config import get_settings
from app.core.supabase import get_supabase
from app.core.async_db import run_db_operation
from app.services.question_generator import QuestionGenerator

# B2: how many prior question stems to load as an exclusion list
_MAX_PRIOR_STEMS = 30

logger = logging.getLogger(__name__)

_supabase = get_supabase()
_question_generator = QuestionGenerator()
_settings = get_settings()

# Sized to the default quiz so all questions generate in a single wave —
# two waves of 5 was the dominant cost of the ~35s users waited.
MAX_CONCURRENT_GENERATIONS = 10
DEFAULT_QUIZ_SIZE = 10

# Wall-clock budget for the whole question-generation job. This is an
# interactive, user-awaited flow (unlike the legacy background pipeline), so
# the cap is tighter: whatever questions complete within the budget are kept
# (partial success beats none), tasks still running past it are cancelled.
QUIZ_GENERATION_TIMEOUT = 150


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _get_concepts_for_course(
    course_id: str,
    document_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return all concepts for a course (or a single document when document_id is given)."""
    def _docs_query():
        q = (
            _supabase.table("documents")
            .select("id")
            .eq("course_id", course_id)
            .eq("status", "completed")
        )
        if document_id:
            q = q.eq("id", document_id)
        return q.execute()

    docs_resp = await run_db_operation(_docs_query)
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


async def _get_prior_question_stems(user_id: str, course_id: str) -> List[str]:
    """Return recent question texts already shown to this user for the course.

    Used as an exclusion list so the generator avoids repeating questions the
    user has already seen.  Capped at _MAX_PRIOR_STEMS to keep prompt size small.
    """
    quiz_resp = await run_db_operation(
        lambda: _supabase.table("course_quizzes")
        .select("id")
        .eq("user_id", user_id)
        .eq("course_id", course_id)
        .execute()
    )
    quiz_ids = [q["id"] for q in (getattr(quiz_resp, "data", None) or [])]
    if not quiz_ids:
        return []

    q_resp = await run_db_operation(
        lambda: _supabase.table("questions")
        .select("question")
        .in_("course_quiz_id", quiz_ids)
        .order("created_at", desc=True)
        .limit(_MAX_PRIOR_STEMS)
        .execute()
    )
    return [r["question"] for r in (getattr(q_resp, "data", None) or [])]


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
        # Haiku: a 40-token title doesn't need Sonnet, and this call runs
        # concurrently with question generation so it must never be the
        # long pole.
        response = await client.messages.create(
            model="claude-haiku-4-5",
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
    excluded_questions: Optional[List[str]] = None,
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
                excluded_questions=excluded_questions or [],
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

async def start_quiz_generation(
    user_id: str,
    course_id: str,
    quiz_size: int = DEFAULT_QUIZ_SIZE,
    document_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate + create the course_quizzes placeholder row, then return
    immediately. The caller (the endpoint) is responsible for scheduling
    `run_quiz_generation_job` as a background task with the returned quiz_id —
    this function never awaits question generation itself, so the HTTP
    request this backs never blocks on Sonnet calls.

    Returns {quiz_id, status, course_id} on success, or {..., error} if there
    are no concepts to generate from at all (a fast, synchronous failure —
    no placeholder row is created in that case).
    """
    logger.info(
        f"Starting quiz generation: user={user_id} course={course_id} size={quiz_size}"
        + (f" document={document_id}" if document_id else "")
    )

    concepts = await _get_concepts_for_course(course_id, document_id)
    if not concepts:
        return {
            "quiz_id": None,
            "status": None,
            "total_questions": 0,
            "course_id": course_id,
            "error": "No concepts found. Upload and process documents first.",
        }

    quiz_id = str(uuid.uuid4())

    # Create the course_quizzes row FIRST so questions can reference it via
    # FK, and so the frontend has something to poll immediately.
    try:
        await run_db_operation(
            lambda: _supabase.table("course_quizzes").insert({
                "id": quiz_id,
                "course_id": course_id,
                "user_id": user_id,
                "name": "Generating…",
                "total_questions": 0,
                "question_order": [],
                "status": "generating",
                "updated_at": _now_iso(),
            }).execute()
        )
    except Exception as e:
        logger.error(f"Failed to create course_quizzes placeholder: {e}")
        return {
            "quiz_id": None,
            "status": None,
            "total_questions": 0,
            "course_id": course_id,
            "error": "Failed to save quiz. Please try again.",
        }

    return {
        "quiz_id": quiz_id,
        "status": "generating",
        "course_id": course_id,
        "concepts": concepts,
    }


async def run_quiz_generation_job(
    quiz_id: str,
    user_id: str,
    course_id: str,
    concepts: List[Dict[str, Any]],
    quiz_size: int,
) -> None:
    """
    The actual generation work, run as a background task after the HTTP
    response for `start_quiz_generation` has already been sent. Bounded by
    QUIZ_GENERATION_TIMEOUT; whatever questions complete within the budget
    are kept (partial success beats none) and course_quizzes.status is always
    left in a terminal state ('completed' or 'failed') — never stuck on
    'generating' past this function returning.

    question_order and total_questions are written once, in the terminal
    update — the frontend waits for a terminal status rather than reading a
    partially-filled row. Invariant: 'failed' ⇔ zero questions; any path that
    dies with partial questions finalizes as 'completed' with what exists.
    """
    name_task: Optional[asyncio.Task] = None
    question_ids: List[str] = []
    pending: set = set()
    try:
        mastery_map = await _get_mastery_map(user_id, course_id)
        selected_concepts = _select_concepts(concepts, mastery_map, quiz_size)
        prior_stems = await _get_prior_question_stems(user_id, course_id)

        # Naming needs only the selected concept names, so it runs
        # concurrently with question generation instead of serially after it.
        concept_names = [c["name"] for c in selected_concepts]
        name_task = asyncio.create_task(_generate_quiz_name(concept_names))

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
        tasks = [
            asyncio.create_task(
                _generate_and_save_question(c, quiz_id, semaphore, prior_stems)
            )
            for c in selected_concepts
        ]

        # All the questions generate in one wave, so a single bounded wait is
        # the whole schedule. (This previously flushed question_order after
        # each completed batch so the frontend could start early; the 3rd of
        # 10 concurrent questions landed ~2s before the last, and the poll
        # interval consumed all of it.)
        done, pending = await asyncio.wait(tasks, timeout=QUIZ_GENERATION_TIMEOUT)

        # Iterate `tasks`, not the `done` set: keeps question order
        # deterministic and matching concept-selection order.
        for t in tasks:
            if t not in done:
                continue
            try:
                qid = t.result()
            except Exception as e:
                logger.error(f"Quiz {quiz_id}: question generation task failed: {e}")
                qid = None
            if qid is not None:
                question_ids.append(qid)

        if pending:
            logger.warning(
                f"Quiz {quiz_id}: {len(pending)}/{len(tasks)} question generation "
                f"task(s) still running past the {QUIZ_GENERATION_TIMEOUT}s budget — "
                f"cancelling, keeping {len(question_ids)} completed."
            )
            for t in pending:
                t.cancel()

        if not question_ids:
            logger.error(f"Quiz {quiz_id}: generation produced no questions")
            name_task.cancel()
            await run_db_operation(
                lambda: _supabase.table("course_quizzes").update({
                    "status": "failed",
                    "error_message": "Could not generate questions. Please try again.",
                    "updated_at": _now_iso(),
                }).eq("id", quiz_id).execute()
            )
            return

        # No shuffle: concept selection is already weighted-random, so task
        # order carries no bias.

        # Usually already resolved by now (questions take far longer than the
        # name). The wait_for guard keeps a hung naming call from stalling the
        # whole job; _generate_quiz_name itself already falls back to "Quiz"
        # on API errors.
        try:
            quiz_name = await asyncio.wait_for(name_task, timeout=30)
        except Exception:
            quiz_name = "Quiz"

        await run_db_operation(
            lambda: _supabase.table("course_quizzes").update({
                "name": quiz_name,
                "total_questions": len(question_ids),
                "question_order": question_ids,
                "status": "completed",
                "updated_at": _now_iso(),
            }).eq("id", quiz_id).execute()
        )
        logger.info(f"Quiz {quiz_id} '{quiz_name}' created with {len(question_ids)} questions")

    except Exception as e:
        logger.error(f"Quiz {quiz_id}: generation job failed: {e}")
        for t in pending:
            t.cancel()
        if name_task is not None and not name_task.done():
            name_task.cancel()
        try:
            if question_ids:
                # Invariant: 'failed' ⇔ zero questions. A crash after partial
                # questions were saved finalizes as completed with what exists.
                quiz_name = "Quiz"
                if name_task is not None and name_task.done() and not name_task.cancelled():
                    try:
                        quiz_name = name_task.result()
                    except Exception:
                        pass
                await run_db_operation(
                    lambda: _supabase.table("course_quizzes").update({
                        "name": quiz_name,
                        "total_questions": len(question_ids),
                        "question_order": question_ids,
                        "status": "completed",
                        "updated_at": _now_iso(),
                    }).eq("id", quiz_id).execute()
                )
                logger.warning(
                    f"Quiz {quiz_id}: finalized as completed with "
                    f"{len(question_ids)} question(s) after job failure"
                )
            else:
                await run_db_operation(
                    lambda: _supabase.table("course_quizzes").update({
                        "status": "failed",
                        "error_message": "Quiz generation failed unexpectedly. Please try again.",
                        "updated_at": _now_iso(),
                    }).eq("id", quiz_id).execute()
                )
        except Exception:
            pass
