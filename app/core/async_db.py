"""
Async database helpers for non-blocking Supabase operations.

Goals:
- Never block the asyncio event loop (Supabase SDK is sync).
- Avoid PostgREST 204 "No Content" / "Missing response" issues by requesting representation.
- Avoid using .select() on builders that don't support it (update/delete in some SDK versions).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, TypeVar, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_db_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="supabase_db_")

T = TypeVar("T")


async def run_db_operation(operation: Callable[[], T]) -> T:
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(_db_executor, operation)
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        raise


def _try_execute(builder) -> Any:
    """
    Centralized execute with a safe fallback when SDK returns 204 or produces 'Missing response'
    errors in some versions. We just re-raise; callers can handle.
    """
    return builder.execute()


async def db_select(supabase, table: str, columns: str = "*", **filters) -> Any:
    def _execute():
        q = supabase.table(table).select(columns)
        for k, v in filters.items():
            q = q.eq(k, v)
        return _try_execute(q)

    return await run_db_operation(_execute)


async def db_select_single(supabase, table: str, columns: str = "*", **filters) -> Any:
    """
    Single row read:
    - maybe_single() prevents PGRST116 when 0 rows.
    """
    def _execute():
        q = supabase.table(table).select(columns)
        for k, v in filters.items():
            q = q.eq(k, v)
        # important
        return _try_execute(q.maybe_single())

    return await run_db_operation(_execute)


async def db_insert(supabase, table: str, data: dict | list, returning: str = "representation") -> Any:
    """
    Insert with returning representation to avoid 204.
    """
    def _execute():
        # Most supabase-py versions support returning=
        try:
            q = supabase.table(table).insert(data, returning=returning)
            return _try_execute(q)
        except TypeError:
            # Older versions: no returning param -> fallback
            q = supabase.table(table).insert(data)
            return _try_execute(q)

    return await run_db_operation(_execute)


async def db_update(supabase, table: str, data: dict, returning: str = "representation", **filters) -> Any:
    """
    Update:
    - DO NOT chain .select() (not supported in some versions)
    - ask for returning representation when available
    """
    def _execute():
        try:
            q = supabase.table(table).update(data, returning=returning)
        except TypeError:
            q = supabase.table(table).update(data)

        for k, v in filters.items():
            q = q.eq(k, v)

        return _try_execute(q)

    return await run_db_operation(_execute)


async def db_upsert(
    supabase,
    table: str,
    data: dict | list,
    on_conflict: Optional[str] = None,
    returning: str = "representation",
) -> Any:
    """
    Upsert:
    - safest path for concurrency
    """
    def _execute():
        kwargs = {}
        if on_conflict:
            kwargs["on_conflict"] = on_conflict

        try:
            q = supabase.table(table).upsert(data, returning=returning, **kwargs)
            return _try_execute(q)
        except TypeError:
            # Older versions: no returning param
            q = supabase.table(table).upsert(data, **kwargs)
            return _try_execute(q)

    return await run_db_operation(_execute)


async def db_delete(supabase, table: str, returning: str = "representation", **filters) -> Any:
    """
    Delete:
    - DO NOT chain .select()
    - ask for returning if possible
    """
    def _execute():
        try:
            q = supabase.table(table).delete(returning=returning)
        except TypeError:
            q = supabase.table(table).delete()

        for k, v in filters.items():
            q = q.eq(k, v)

        return _try_execute(q)

    return await run_db_operation(_execute)


async def db_storage_download(supabase, bucket: str, path: str) -> bytes:
    def _execute():
        return supabase.storage.from_(bucket).download(path)

    return await run_db_operation(_execute)


def shutdown_db_executor():
    _db_executor.shutdown(wait=True)
