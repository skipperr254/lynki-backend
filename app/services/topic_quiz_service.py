"""
Topic Quiz Service

Generates fresh, on-demand quizzes for a specific topic using Claude Sonnet.
Questions are stored as JSONB in topic_quiz_sessions — separate from the BKT
question bank — so they never interfere with adaptive study sessions.

Session lifecycle:
  - in_progress: user has started but not finished
  - completed: all questions answered; next click generates a fresh set
"""
from __future__ import annotations

import logging
import uuid
import random
from datetime import datetime, timezone
from typing import Any

from app.core.supabase import get_supabase
from app.services.question_generator import QuestionGenerator
from app.services.bkt.service import BKTService

logger = logging.getLogger(__name__)

# Questions to generate based on topic size (# concepts)
_Q_SMALL = 5   # <= 2 concepts
_Q_MEDIUM = 7  # 3–5 concepts
_Q_LARGE = 10  # > 5 concepts
_MAX_PER_CONCEPT = 3


def _target_count(num_concepts: int) -> int:
    if num_concepts <= 2:
        return _Q_SMALL
    if num_concepts <= 5:
        return _Q_MEDIUM
    return _Q_LARGE


def _distribute(num_concepts: int, total: int) -> list[int]:
    """Distribute `total` questions across `num_concepts` concepts (round-robin, cap per concept)."""
    per = [0] * num_concepts
    idx = 0
    for _ in range(total):
        # find next concept that hasn't hit the cap
        start = idx
        while per[idx % num_concepts] >= _MAX_PER_CONCEPT:
            idx += 1
            if idx - start >= num_concepts:
                break  # all capped, stop
        per[idx % num_concepts] += 1
        idx += 1
    return per


def _strip_correct(question: dict) -> dict:
    """Return a copy of the question with is_correct removed from all options."""
    q = dict(question)
    q["options"] = [
        {k: v for k, v in opt.items() if k != "is_correct"}
        for opt in question["options"]
    ]
    return q




class TopicQuizService:

    @staticmethod
    async def get_or_create_session(
        user_id: str,
        course_id: str,
        topic_id: str,
        question_format: str = "standard",
    ) -> dict[str, Any]:
        """
        Return an active (in_progress) session if one exists, otherwise generate
        a new quiz and return the new session.
        Returns the session dict with questions stripped of is_correct.
        """
        db = get_supabase()

        # Check for an existing in_progress session
        existing = (
            db.table("topic_quiz_sessions")
            .select("*")
            .eq("user_id", user_id)
            .eq("topic_id", topic_id)
            .eq("status", "in_progress")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if existing.data:
            session = existing.data[0]
            return _public_session(session)

        # No active session — generate a new one
        return await TopicQuizService._generate_session(
            db, user_id, course_id, topic_id, question_format
        )

    @staticmethod
    async def _generate_session(
        db,
        user_id: str,
        course_id: str,
        topic_id: str,
        question_format: str,
    ) -> dict[str, Any]:
        # 1. Fetch topic name
        topic_row = db.table("topics").select("name").eq("id", topic_id).single().execute()
        if not topic_row.data:
            raise ValueError(f"Topic {topic_id} not found")
        topic_name = topic_row.data["name"]

        # 2. Fetch all concepts for this topic
        concepts_res = (
            db.table("concepts")
            .select("id, name, explanation, source_text")
            .eq("topic_id", topic_id)
            .execute()
        )
        concepts = concepts_res.data or []
        if not concepts:
            raise ValueError(f"No concepts found for topic {topic_id}")

        num_concepts = len(concepts)
        total_target = _target_count(num_concepts)
        distribution = _distribute(num_concepts, total_target)

        # 3. Generate questions per concept in parallel-ish (sequential is fine for small counts)
        generator = QuestionGenerator()
        all_questions: list[dict] = []

        for concept, num_q in zip(concepts, distribution):
            if num_q == 0:
                continue
            generated = await generator.generate_questions_for_concept(
                concept_id=concept["id"],
                concept_name=concept["name"],
                concept_explanation=concept.get("explanation") or "",
                source_text=concept.get("source_text") or "",
                num_questions=num_q,
                question_format=question_format,
            )
            for gq in generated:
                # Shuffle options so the correct answer isn't always first
                # (Claude follows the prompt template which puts correct first)
                options = [
                    {
                        "text": opt.option_text,
                        "is_correct": opt.is_correct,
                        "explanation": opt.explanation,
                    }
                    for opt in gq.options
                ]
                random.shuffle(options)
                # Re-assign sequential indices after shuffling
                for idx, opt in enumerate(options):
                    opt["index"] = idx

                all_questions.append({
                    "id": str(uuid.uuid4()),
                    "question": gq.question,
                    "concept_id": gq.concept_id,
                    "concept_name": concept["name"],
                    "difficulty": gq.difficulty_level,
                    "hint": gq.hint,
                    "options": options,
                    "question_format": gq.question_format,
                    "post_answer_summary": gq.post_answer_summary,
                })

        if not all_questions:
            raise ValueError("Question generation produced no questions")

        random.shuffle(all_questions)

        # 4. Persist session
        now = datetime.now(timezone.utc).isoformat()
        row = {
            "user_id": user_id,
            "course_id": course_id,
            "topic_id": topic_id,
            "topic_name": topic_name,
            "status": "in_progress",
            "questions": all_questions,
            "answers": [],
            "current_index": 0,
            "total_questions": len(all_questions),
            "correct_count": 0,
            "created_at": now,
            "updated_at": now,
        }
        inserted = db.table("topic_quiz_sessions").insert(row).execute()
        session = inserted.data[0]
        logger.info(
            f"Created topic quiz session {session['id']} "
            f"({len(all_questions)} questions) for topic {topic_id}"
        )
        return _public_session(session)

    @staticmethod
    async def check_answer(
        session_id: str,
        question_index: int,
        selected_option: int,
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        """
        Fast path: fetch session, check correctness, return result.
        Does NOT write anything — call persist_answer in a background task.
        Returns (result_for_client, session_snapshot, is_correct).
        """
        db = get_supabase()

        session_res = (
            db.table("topic_quiz_sessions")
            .select("*")
            .eq("id", session_id)
            .single()
            .execute()
        )
        if not session_res.data:
            raise ValueError(f"Session {session_id} not found")
        session = session_res.data

        questions: list[dict] = session["questions"]
        if question_index >= len(questions):
            raise ValueError("question_index out of range")

        question = questions[question_index]
        options = question["options"]

        correct_opt = next((o for o in options if o["is_correct"]), None)
        selected_opt = next((o for o in options if o["index"] == selected_option), None)
        is_correct = selected_opt is not None and selected_opt.get("is_correct", False)

        result = {
            "is_correct": is_correct,
            "correct_option_index": correct_opt["index"] if correct_opt else 0,
            "correct_option_text": correct_opt["text"] if correct_opt else "",
            "selected_explanation": selected_opt["explanation"] if selected_opt else "",
            "correct_explanation": correct_opt["explanation"] if correct_opt else "",
            "hint": question.get("hint"),
        }
        return result, session, is_correct

    @staticmethod
    async def persist_answer(
        session: dict[str, Any],
        question_index: int,
        selected_option: int,
        is_correct: bool,
    ) -> None:
        """
        Background task: write the answer record + update BKT mastery.
        Safe to call after the HTTP response is already sent.
        """
        db = get_supabase()

        answers: list[dict] = session.get("answers") or []
        already = any(a["question_index"] == question_index for a in answers)
        if already:
            return  # duplicate submission, nothing to do

        answers.append({
            "question_index": question_index,
            "selected_option": selected_option,
            "is_correct": is_correct,
        })
        new_correct = session["correct_count"] + (1 if is_correct else 0)
        new_index = max(session["current_index"], question_index + 1)

        db.table("topic_quiz_sessions").update({
            "answers": answers,
            "correct_count": new_correct,
            "current_index": new_index,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", session["id"]).execute()

        concept_id = session["questions"][question_index].get("concept_id")
        if concept_id:
            try:
                await BKTService.update_mastery_for_concept(
                    user_id=session["user_id"],
                    course_id=session["course_id"],
                    concept_id=concept_id,
                    is_correct=is_correct,
                )
            except Exception as e:
                logger.warning(f"BKT update failed for concept {concept_id}: {e}")

    @staticmethod
    async def complete_session(session_id: str) -> None:
        """Mark a session as completed."""
        db = get_supabase()
        now = datetime.now(timezone.utc).isoformat()
        db.table("topic_quiz_sessions").update({
            "status": "completed",
            "completed_at": now,
            "updated_at": now,
        }).eq("id", session_id).execute()


def _public_session(session: dict) -> dict:
    """Strip is_correct from questions before sending to client."""
    return {
        "id": session["id"],
        "topic_id": session["topic_id"],
        "topic_name": session["topic_name"],
        "status": session["status"],
        "current_index": session["current_index"],
        "total_questions": session["total_questions"],
        "correct_count": session["correct_count"],
        "questions": [_strip_correct(q) for q in session["questions"]],
        "created_at": session["created_at"],
    }
