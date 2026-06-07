"""
Watchdog: periodically marks documents stuck in pending/processing as failed.

Design choice: straight-to-failed (not re-enqueue). The Claude API already
retries 3× per chunk during active processing, so transient failures are
covered. If the Render worker dies mid-task the built-in asyncio.wait_for
timeout never fires — this watchdog catches that case.

Threshold: 15 minutes (above the 10-min DOCUMENT_PROCESSING_TIMEOUT so the
built-in timeout wins for active tasks and the watchdog only fires for dead ones).
Interval: every 5 minutes.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from app.core.supabase import get_supabase
from app.core.async_db import run_db_operation

logger = logging.getLogger(__name__)

STUCK_THRESHOLD_MINUTES = 15
WATCHDOG_INTERVAL_SECONDS = 300

_STUCK_MESSAGE = (
    "Processing timed out. Please use the Reprocess button to try again."
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


async def watchdog_loop() -> None:
    """Background task: sweep immediately on startup, then every WATCHDOG_INTERVAL_SECONDS."""
    await mark_stuck_documents_failed()
    while True:
        await asyncio.sleep(WATCHDOG_INTERVAL_SECONDS)
        await mark_stuck_documents_failed()
