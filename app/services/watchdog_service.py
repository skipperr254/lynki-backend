"""
Watchdog: periodically marks documents stuck in pending/processing as failed,
and finalizes course_quizzes rows stuck in 'generating'.

Documents: straight-to-failed (not re-enqueue). The Claude API already
retries 3x per chunk during active processing, so transient failures are
covered. If the Render worker dies mid-task the built-in asyncio.wait_for /
asyncio.wait timeout never fires — this watchdog catches that case.

Quizzes: a stale 'generating' row with questions represents a usable partial
quiz, so those are finalized as 'completed' with what exists; only rows with
zero questions become 'failed' (invariant: 'failed' ⇔ zero questions). A live
job writes question_order once, at the end, and finishes inside its own 150s
budget — comfortably under the 5-minute staleness threshold below — so it
never trips this sweep.

Documents threshold: 15 minutes (above the 10-min DOCUMENT_PROCESSING_TIMEOUT
so the built-in timeout wins for active tasks and the watchdog only fires for
dead ones). Quizzes threshold: 5 minutes (above the on-demand flow's own
150s QUIZ_GENERATION_TIMEOUT, same reasoning, tighter because generation is a
much shorter interactive job). Interval: every 5 minutes.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from app.core.supabase import get_supabase
from app.core.async_db import run_db_operation

logger = logging.getLogger(__name__)

STUCK_THRESHOLD_MINUTES = 15
STUCK_QUIZ_THRESHOLD_MINUTES = 5
WATCHDOG_INTERVAL_SECONDS = 300

_STUCK_MESSAGE = (
    "Processing timed out. Please use the Reprocess button to try again."
)
_STUCK_QUIZ_MESSAGE = (
    "Generation timed out. Please try generating the quiz again."
)


async def mark_stuck_documents_failed() -> int:
    """Mark pending/processing docs not updated in >STUCK_THRESHOLD_MINUTES as failed."""
    cutoff = (
        datetime.now(timezone.utc) - timedelta(minutes=STUCK_THRESHOLD_MINUTES)
    ).isoformat()
    try:
        resp = await run_db_operation(
            lambda: get_supabase()
            .table("documents")
            .update({"status": "failed", "error_message": _STUCK_MESSAGE})
            .in_("status", ["pending", "processing"])
            .lt("updated_at", cutoff)
            .execute()
        )
        count = len(resp.data) if resp and resp.data else 0
        if count:
            logger.warning("Watchdog: marked %d stuck document(s) as failed", count)
        return count
    except Exception:
        logger.exception("Watchdog: error during stuck-document sweep")
        return 0


async def finalize_stuck_quizzes() -> int:
    """Finalize course_quizzes rows stuck in 'generating' (worker died mid-job).

    Rows with questions (non-empty question_order) → 'completed' with what
    exists; rows with none → 'failed'. See module docstring.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(minutes=STUCK_QUIZ_THRESHOLD_MINUTES)
    ).isoformat()
    try:
        resp = await run_db_operation(
            lambda: get_supabase()
            .table("course_quizzes")
            .select("id, name, question_order")
            .eq("status", "generating")
            .lt("updated_at", cutoff)
            .execute()
        )
        rows = (resp.data if resp else None) or []
        count = 0
        for row in rows:
            order = row.get("question_order") or []
            now = datetime.now(timezone.utc).isoformat()
            if order:
                payload = {
                    "status": "completed",
                    "total_questions": len(order),
                    "updated_at": now,
                }
                # The placeholder name is "Generating…" (Unicode ellipsis) —
                # startswith, not equality. Replace it if the job died before
                # the real name was written.
                if (row.get("name") or "").startswith("Generating"):
                    payload["name"] = "Quiz"
            else:
                payload = {
                    "status": "failed",
                    "error_message": _STUCK_QUIZ_MESSAGE,
                    "updated_at": now,
                }
            try:
                await run_db_operation(
                    lambda p=payload, rid=row["id"]: get_supabase()
                    .table("course_quizzes")
                    .update(p)
                    .eq("id", rid)
                    # A live job's terminal write racing this sweep wins.
                    .eq("status", "generating")
                    .execute()
                )
                count += 1
            except Exception:
                logger.exception("Watchdog: failed to finalize quiz %s", row.get("id"))
        if count:
            logger.warning("Watchdog: finalized %d stuck quiz(zes)", count)
        return count
    except Exception:
        logger.exception("Watchdog: error during stuck-quiz sweep")
        return 0


async def watchdog_loop() -> None:
    """Background task: sweep immediately on startup, then every WATCHDOG_INTERVAL_SECONDS."""
    await mark_stuck_documents_failed()
    await finalize_stuck_quizzes()
    while True:
        await asyncio.sleep(WATCHDOG_INTERVAL_SECONDS)
        await mark_stuck_documents_failed()
        await finalize_stuck_quizzes()
