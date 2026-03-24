"""
Study Plan Service

Fetches BKT progress + pass probability, asks Claude Haiku to produce a
structured JSON study plan (sessions + activities), enriches each activity with
concept_id/topic_id by matching names against the BKT data, and upserts the
result into the study_plans table.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from app.core.config import get_settings
from app.core.supabase import get_supabase
from app.core.async_db import db_select_single, db_upsert
from app.services.bkt.service import BKTService
from app.services.test_service import get_pass_chance

settings = get_settings()
logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 2000
CLAUDE_TEMPERATURE = 0.4
CLAUDE_TIMEOUT_SECONDS = 60

_supabase = get_supabase()
_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)


async def generate_study_plan(user_id: str, course_id: str) -> Dict[str, Any]:
    """
    Orchestrates study plan generation.
    Returns {"plan_json": dict, "generated_at": str (ISO-8601)}.
    """
    progress, pass_chance_data, course = await asyncio.gather(
        BKTService.get_course_progress(user_id, course_id),
        get_pass_chance(user_id, course_id),
        db_select_single(_supabase, "courses", "title,test_date,target_grade", id=course_id),
    )

    course_meta = getattr(course, "data", None) or {}

    prompt_context = _build_prompt_context(progress, pass_chance_data, course_meta)
    raw_json_str = await _call_claude(prompt_context)

    plan_dict = _parse_and_enrich(raw_json_str, progress)

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


def _parse_and_enrich(raw: str, progress: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse Claude's JSON output, strip any accidental markdown fences,
    then enrich each activity with concept_id and topic_id.
    """
    # Strip ```json ... ``` fences if Claude added them
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Claude JSON: %s\nRaw: %s", e, raw[:500])
        raise ValueError("Study plan generation produced invalid JSON. Please try again.")

    # Build name → (concept_id, topic_id) map (case-insensitive)
    concept_map: Dict[str, tuple] = {}
    for topic in progress.get("topics", []):
        tid = topic.get("topic_id", "")
        for concept in topic.get("concepts", []):
            name_key = (concept.get("concept_name") or "").strip().lower()
            if name_key:
                concept_map[name_key] = (concept.get("concept_id", ""), tid)

    for session in plan.get("sessions", []):
        for activity in session.get("activities", []):
            name_key = (activity.get("concept_name") or "").strip().lower()
            cid, tid = concept_map.get(name_key, ("", ""))
            activity["concept_id"] = cid
            activity["topic_id"] = tid

    return plan


def _build_prompt_context(
    progress: Dict[str, Any],
    pass_chance_data: Dict[str, Any],
    course: Dict[str, Any],
) -> str:
    course_title = course.get("title", "this course")
    test_date_str: Optional[str] = course.get("test_date")
    target_grade_raw: float = float(course.get("target_grade") or 1.0)
    target_pct = round(target_grade_raw * 100)

    days_remaining = _compute_days_remaining(test_date_str)

    total: int = progress.get("total_concepts", 0)
    mastered: int = progress.get("mastered_concepts", 0)
    overall_pct: int = progress.get("overall_progress", 0)

    pass_prob = pass_chance_data.get("pass_probability")
    pass_pct_str = f"{round(float(pass_prob) * 100)}%" if pass_prob is not None else "unknown (no quiz attempts yet)"

    unmastered_rows: List[tuple] = []  # (p_mastery_pct, line)
    not_started_lines: List[str] = []
    mastered_names: List[str] = []

    for topic in progress.get("topics", []):
        topic_name = topic.get("topic_name", "Unknown Topic")
        for concept in topic.get("concepts", []):
            name = concept.get("concept_name", "?")
            p = float(concept.get("p_mastery", 0.2))
            attempts = int(concept.get("n_attempts", 0))
            is_mastered = concept.get("is_mastered", False)

            if is_mastered:
                mastered_names.append(name)
            elif attempts == 0:
                not_started_lines.append(f"- {name} | topic: {topic_name}")
            else:
                pct = round(p * 100)
                unmastered_rows.append(
                    (pct, f"- {name} | topic: {topic_name} | mastery: {pct}% | attempts: {attempts}")
                )

    unmastered_rows.sort(key=lambda x: x[0])
    sorted_unmastered = [line for _, line in unmastered_rows]

    parts = [
        f"COURSE: {course_title}",
        f"EXAM DATE: {test_date_str or 'not set'} ({days_remaining} days from today)",
        f"TARGET GRADE: {target_pct}%",
        f"CURRENT PASS PROBABILITY: {pass_pct_str}",
        f"OVERALL MASTERY: {mastered}/{total} concepts mastered ({overall_pct}%)",
        "",
    ]

    if sorted_unmastered:
        parts.append("CONCEPTS NEEDING WORK (weakest first):")
        parts.extend(sorted_unmastered)
        parts.append("")

    if not_started_lines:
        parts.append("CONCEPTS NOT YET STARTED:")
        parts.extend(not_started_lines)
        parts.append("")

    if mastered_names:
        shown = mastered_names[:10]
        suffix = " and more..." if len(mastered_names) > 10 else ""
        parts.append(f"ALREADY MASTERED ({len(mastered_names)} concepts): {', '.join(shown)}{suffix}")
        parts.append("")

    return "\n".join(parts)


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
        "Here is the student's current progress data:\n\n"
        f"{prompt_context}\n"
        "Generate their personalised study plan as JSON now."
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
        "You are a warm, encouraging study coach for a garden-themed learning app called Lynki.\n"
        "Generate a personalised study plan as a structured JSON object.\n\n"
        "TONE: Warm, encouraging, garden-language (tend, grow, bloom, seeds). Speak directly to the student as 'you'.\n\n"
        "OUTPUT: Return ONLY a valid JSON object — no markdown fences, no explanation, no preamble.\n\n"
        "EXACT SCHEMA:\n"
        "{\n"
        '  "overview": "2-3 sentence honest assessment. Mention pass probability and mastery count. If all mastered, celebrate and suggest review.",\n'
        '  "sessions": [\n'
        "    {\n"
        '      "label": "Day 1",\n'
        '      "theme": "Short theme (3-5 words)",\n'
        '      "activities": [\n'
        "        {\n"
        '          "concept_name": "EXACT concept name from the data",\n'
        '          "topic_name": "EXACT topic name from the data",\n'
        '          "guidance": "One specific, actionable study sentence for this concept"\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "tip": "One memorable piece of encouragement or exam-day advice (2-3 sentences)."\n'
        "}\n\n"
        "SESSION RULES:\n"
        "- 1 session per day if ≤7 days remaining\n"
        "- 1 session per 2 days if 8-14 days remaining\n"
        "- 1 session per week if >14 days remaining\n"
        "- 2-4 activities per session (manageable, not overwhelming)\n"
        "- Distribute weakest concepts first, across sessions\n"
        "- Do NOT include already-mastered concepts in activities\n"
        "- If all concepts are mastered, create 1-2 review sessions with the most important concepts\n\n"
        "CRITICAL: Use the EXACT concept_name and topic_name strings as they appear in the data provided. "
        "Do not paraphrase, abbreviate, or alter them in any way."
    )
