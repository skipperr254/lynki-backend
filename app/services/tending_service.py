import json
import logging
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from anthropic import AsyncAnthropic, APITimeoutError, APIConnectionError
from app.core.config import get_settings
from app.core.supabase import get_supabase
from app.core.async_db import db_select, db_insert, db_update, run_db_operation

settings = get_settings()
logger = logging.getLogger(__name__)

TENDING_SYSTEM_PROMPT = """
You are a Learning Scientist and memory expert specialized in active recall and spaced repetition.
Your goal is to transform a list of educational concepts into a multi-stage study session.

STAGES TO GENERATE:
1. Recall Cards: 4-6 cards. Front is a question/prompt, Back is the core answer. Focus on foundational definitions.
2. Active Recall: A high-level prompt asking the student to explain the relationship between 2-3 key concepts in their own words. Provide a 'source_paragraph' (80-120 words) that contains the perfect explanation for comparison.
3. Mnemonics: 3 creative mnemonics for the most difficult concepts. Use catchy hooks and clear explanations.
4. Connections: 4-5 pairs of matching items (e.g., Concept -> Real world analogy, or Term -> Function).

OUTPUT FORMAT:
You must return a valid JSON object with the following structure:
{
  "recall_cards": {
    "stage": "recall_cards",
    "cards": [
      { "id": "uuid", "front": "...", "back": "..." }
    ]
  },
  "active_recall": {
    "stage": "active_recall",
    "prompt": "...",
    "source_paragraph": "..."
  },
  "mnemonics": {
    "stage": "mnemonics",
    "mnemonics": [
      { "id": "uuid", "hook": "...", "explanation": "..." }
    ]
  },
  "connections": {
    "stage": "connections",
    "type": "analogy",
    "pairs": [
      { "id": "uuid", "left": "...", "right": "..." }
    ]
  }
}

Use the provided concept data strictly. Do not hallucinate external facts.
Keep the IDs as random UUID strings.
"""

EVALUATION_SYSTEM_PROMPT = """
You are an expert tutor grading a student's active-recall explanation.
Compare the Student Response to the Source Paragraph.

IDENTIFY:
1. 'got_right': List of specific concepts or facts from the source paragraph correctly explained by the student.
2. 'missed': List of key concepts or facts from the source paragraph that the student forgot or explained incorrectly.

Output must be JSON:
{
  "got_right": ["Fact A", "Fact B"],
  "missed": ["Fact C"],
  "score": 0.85 
}
'score' should be between 0 and 1.
"""

# Wall-clock budgets for the two Claude calls, and a cap on the SDK's own
# silent retries. Without these the client defaults apply — a 600s timeout and
# max_retries=2 — so a single 429/529 turned one generation into three
# sequential 2500-token Sonnet calls with backoff and no log line, which is
# what made /generate take minutes. Every other service in this app caps both;
# see question_generator.CLAUDE_TIMEOUT_SECONDS. The generate budget is set
# below the frontend's fetch timeout so the server always loses the race and
# returns a real error instead of the client aborting a live request.
GENERATE_TIMEOUT_SECONDS = 75
EVALUATE_TIMEOUT_SECONDS = 45
MAX_API_RETRIES = 1


def _parse_json_response(response_text: str) -> Dict[str, Any]:
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}') + 1
    if start_idx == -1 or end_idx == 0:
        raise ValueError("No JSON object found in response")
    json_str = response_text[start_idx:end_idx]
    return json.loads(json_str)


class TendingService:
    def __init__(self):
        self.client = AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            max_retries=MAX_API_RETRIES,
        )
        self.model = "claude-sonnet-4-6"  # or settings.CLAUDE_MODEL

    async def generate_session(self, user_id: str, course_id: str, topic_id: str) -> Dict[str, Any]:
        db = get_supabase()

        # Every Supabase call here goes through run_db_operation. The sync SDK
        # blocks whichever thread it runs on, and calling it bare inside an
        # `async def` blocks the whole event loop — freezing every other
        # in-flight request on the worker, not just this one.

        # 1. Fetch Topic Title
        topic_res = await run_db_operation(
            lambda: db.table("topics").select("name").eq("id", topic_id).single().execute()
        )
        topic_title = topic_res.data["name"] if topic_res.data else "Unknown Topic"

        # 2. Fetch Concepts (limit to 10 for prompt size)
        concepts_res = await run_db_operation(
            lambda: db.table("concepts")
            .select("name, explanation")
            .eq("topic_id", topic_id)
            .limit(10)
            .execute()
        )
        concepts = concepts_res.data or []

        concept_context = "\\n".join([f"- {c['name']}: {c['explanation']}" for c in concepts])

        # 3. Call Claude
        try:
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=2500,
                    system=TENDING_SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": f"Topic: {topic_title}\\nConcepts:\\n{concept_context}"}
                    ]
                ),
                timeout=GENERATE_TIMEOUT_SECONDS,
            )
            content = response.content[0].text
            generated_data = _parse_json_response(content)
            # Fold the title into generated_content so a later resume (which
            # hydrates straight from this row) doesn't need a topics join.
            generated_data["topic_title"] = topic_title
        except (asyncio.TimeoutError, APITimeoutError, APIConnectionError) as e:
            logger.error(
                f"Claude tending generation timed out after {GENERATE_TIMEOUT_SECONDS}s: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Claude tending generation failed: {e}")
            raise

        # 4. Save Session to DB
        # Invariant: at most one incomplete (not completed, not abandoned) row
        # per user+topic. Enforced here rather than relied upon client-side —
        # if anything upstream ever races and calls generate twice for the
        # same topic (a duplicate frontend re-render, two tabs, a retry),
        # this stops it from leaving orphaned rows that a later "resume"
        # lookup could pick up instead of the session actually in use.
        await run_db_operation(
            lambda: db.table("topic_tending_sessions").update({
                "abandoned_at": datetime.now(timezone.utc).isoformat(),
            }).eq("user_id", user_id).eq("topic_id", topic_id).is_(
                "completed_at", "null"
            ).is_("abandoned_at", "null").execute()
        )

        session_id = str(uuid.uuid4())
        session_row = {
            "id": session_id,
            "user_id": user_id,
            "course_id": course_id,
            "topic_id": topic_id,
            "current_step": "recall_cards",
            "generated_content": generated_data
        }
        
        await db_insert(db, "topic_tending_sessions", session_row)

        return {
            "session_id": session_id,
            "course_id": course_id,
            "topic_id": topic_id,
            "topic_title": topic_title,
            **generated_data
        }

    async def evaluate_recall(self, session_id: str, student_response: str) -> Dict[str, Any]:
        db = get_supabase()
        
        # 1. Fetch Session for source paragraph
        session_res = await run_db_operation(
            lambda: db.table("topic_tending_sessions")
            .select("generated_content")
            .eq("id", session_id)
            .single()
            .execute()
        )
        if not session_res.data:
            raise ValueError("Session not found")
        
        session_data = session_res.data["generated_content"]
        source_paragraph = session_data.get("active_recall", {}).get("source_paragraph", "")

        # 2. Call Claude for evaluation
        try:
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    system=EVALUATION_SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": f"Source Paragraph: {source_paragraph}\\nStudent Response: {student_response}"}
                    ]
                ),
                timeout=EVALUATE_TIMEOUT_SECONDS,
            )
            content = response.content[0].text
            evaluation = _parse_json_response(content)
        except (asyncio.TimeoutError, APITimeoutError, APIConnectionError) as e:
            logger.error(
                f"Claude recall evaluation timed out after {EVALUATE_TIMEOUT_SECONDS}s: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Claude recall evaluation failed: {e}")
            raise

        return {
            "got_right": evaluation.get("got_right", []),
            "missed": evaluation.get("missed", []),
            "source_paragraph": source_paragraph,
            "score": evaluation.get("score", 0.0)
        }
