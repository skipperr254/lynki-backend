"""
Study Plan Service

Fetches BKT progress + pass probability + quiz history, then asks Claude Sonnet to
produce a markdown "growth guide" following the garden-metaphor spec. The result is
stored in the study_plans table as plan_json = {"markdown": "...", "version": 2}.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from app.core.config import get_settings
from app.core.supabase import get_supabase
from app.core.async_db import db_select_single, db_upsert
from app.services.bkt.service import BKTService
from app.services.test_service import get_pass_chance, get_test_history

settings = get_settings()
logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS = 3000
CLAUDE_TEMPERATURE = 0.7
CLAUDE_TIMEOUT_SECONDS = 90

_supabase = get_supabase()
_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)


async def generate_study_plan(user_id: str, course_id: str) -> Dict[str, Any]:
    """
    Orchestrates study plan generation.
    Returns {"plan_json": dict, "generated_at": str (ISO-8601)}.
    """
    progress, pass_chance_data, course, quiz_history = await asyncio.gather(
        BKTService.get_course_progress(user_id, course_id),
        get_pass_chance(user_id, course_id),
        db_select_single(_supabase, "courses", "title,test_date,target_grade", id=course_id),
        get_test_history(user_id, course_id, limit=10),
    )

    course_meta = getattr(course, "data", None) or {}

    prompt_context = _build_prompt_context(progress, pass_chance_data, course_meta, quiz_history)
    markdown_str = await _call_claude(prompt_context)

    plan_dict = _wrap_markdown(markdown_str)

    generated_at = datetime.now(timezone.utc).isoformat()
    await db_upsert(
        _supabase,
        "study_plans",
        {
            "user_id": user_id,
            "course_id": course_id,
            "plan_json": plan_dict,
            "generated_at": generated_at,
        },
        on_conflict="user_id,course_id",
    )

    return {"plan_json": plan_dict, "generated_at": generated_at}


def _wrap_markdown(text: str) -> Dict[str, Any]:
    """
    Wrap the raw markdown string in a versioned dict for storage.
    Strips any accidental code fences Claude may have added.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        stripped = "\n".join(inner).strip()
    return {"markdown": stripped, "version": 2}


def _build_prompt_context(
    progress: Dict[str, Any],
    pass_chance_data: Dict[str, Any],
    course: Dict[str, Any],
    quiz_history: List[Dict[str, Any]],
) -> str:
    course_name = course.get("title", "this course")
    exam_date: Optional[str] = course.get("test_date")
    target_grade_raw: float = float(course.get("target_grade") or 1.0)
    target_mastery = round(target_grade_raw * 100)

    days_until_exam = _compute_days_remaining(exam_date)
    current_mastery: int = progress.get("overall_progress", 0)

    # Build topic breakdown with topic-level mastery and knowledge components
    topic_lines: List[str] = []
    for topic in progress.get("topics", []):
        topic_name = topic.get("topic_name", "Unknown Topic")
        concepts = topic.get("concepts", [])

        if not concepts:
            continue

        # Aggregate topic-level mastery from concept p_mastery values
        mastery_values = [float(c.get("p_mastery", 0.2)) for c in concepts]
        topic_mastery_pct = round((sum(mastery_values) / len(mastery_values)) * 100)

        topic_lines.append(f"\n{topic_name} — {topic_mastery_pct}% mastery")
        for concept in concepts:
            kc_name = concept.get("concept_name", "?")
            kc_mastery = round(float(concept.get("p_mastery", 0.2)) * 100)
            topic_lines.append(f"  - {kc_name}: {kc_mastery}%")

    topics_section = "\n".join(topic_lines) if topic_lines else "No topics found for this course."

    # Build quiz history section
    completed_sessions = [s for s in (quiz_history or []) if s.get("status") == "completed"]
    if completed_sessions:
        history_lines = []
        for session in completed_sessions[:8]:
            raw_date = session.get("created_at", "")
            date_str = raw_date[:10] if raw_date else "unknown date"
            score = session.get("correct_count", 0)
            total = session.get("total_questions", 0)
            history_lines.append(f"- {date_str}: {score}/{total} (General Practice)")
        quiz_section = "\n".join(history_lines)
    else:
        quiz_section = "No quiz history yet."

    exam_date_display = exam_date or "not set"

    return (
        f"Course: {course_name}\n"
        f"Exam date: {exam_date_display} ({days_until_exam} days away)\n"
        f"Current overall mastery: {current_mastery}%\n"
        f"Target: {target_mastery}%\n"
        f"\nTopic breakdown:\n{topics_section}\n"
        f"\nRecent quiz history:\n{quiz_section}\n"
    )


def _compute_days_remaining(test_date_str: Optional[str]) -> int:
    if not test_date_str:
        return 0
    try:
        exam = datetime.fromisoformat(test_date_str.split("T")[0])
        today = datetime.now(timezone.utc).date()
        return max(0, (exam.date() - today).days)
    except (ValueError, AttributeError):
        return 0


async def _call_claude(prompt_context: str) -> str:
    system_prompt = _build_system_prompt()
    user_message = (
        "Generate a personalized study plan for this student.\n\n"
        f"{prompt_context}\n"
        "Write a study plan that:\n"
        "1. Identifies which topics need the most water (biggest gap from target)\n"
        "2. Gives 2-3 specific actions per weak topic with time estimates\n"
        "3. Suggests a realistic daily/weekly rhythm based on days until exam\n"
        "4. Shows projected growth if the student follows the plan"
    )

    response = await asyncio.wait_for(
        _client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=CLAUDE_TEMPERATURE,
        ),
        timeout=CLAUDE_TIMEOUT_SECONDS,
    )
    return response.content[0].text


def _build_system_prompt() -> str:
    return (
        "You are a study plan generator for PassAI, an adaptive study app for IB/AP/GCSE students.\n\n"
        "You write study plans using a garden growth metaphor. You never sound clinical, robotic, "
        "or like a task list. You sound like a warm, experienced teacher who knows the student can do this.\n\n"
        "Rules:\n"
        '- Address the student directly ("you", not "the student")\n'
        "- Start with their strongest area first to build confidence, then address weak areas\n"
        '- Use garden language: "water", "tend", "nurture", "bloom", "grow", "roots"\n'
        '- Never say "you\'re behind", "needs attention", "more practice needed", or "high priority"\n'
        '- Give specific time estimates for each recommendation (e.g. "15 minutes")\n'
        '- Show the projected outcome if they follow the plan (e.g. "this will grow from 45% to 65%")\n'
        "- Sort weak areas by biggest gap from target — those get the most attention\n"
        "- Keep each topic section to 2-3 concrete actions, not a wall of tasks\n"
        "- End with an encouraging line that references their exam date naturally\n"
        "- Output as markdown\n\n"
        "Use this structure for your markdown output:\n"
        "- A title line: `# Your Growth Guide — [Course Name]`\n"
        "- An intro paragraph (2-3 sentences, garden tone, mentions days until exam)\n"
        "- One `## Topic Name — XX%` section per topic (strongest first, then weakest)\n"
        "  - A one-sentence description of where this topic stands (garden metaphor)\n"
        "  - 2-3 bullet actions with time estimates: `- 15 min: [action]`\n"
        "  - A projected growth line: `- After this: this grows from XX% to YY%`\n"
        "- A `## Your rhythm:` section with a 2-3 sentence cadence suggestion\n"
        "- A closing encouragement paragraph that naturally mentions the exam date\n\n"
        "Do NOT include: task numbers, priority labels, checklists, or clinical progress language."
    )
