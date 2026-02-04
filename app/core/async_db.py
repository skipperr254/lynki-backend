"""
Async database helpers for non-blocking Supabase operations.

The Supabase Python SDK uses synchronous HTTP calls which block the asyncio event loop.
This module provides async wrappers that run database operations in a thread pool executor,
preventing event loop blocking and allowing concurrent request handling.

IMPORTANT: Always use these helpers instead of direct .execute() calls in async functions.
"""

import asyncio
import logging
from functools import partial
from typing import TypeVar, Callable, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Dedicated thread pool for database operations
# Using a separate pool prevents database I/O from competing with other async tasks
_db_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="supabase_db_")

T = TypeVar('T')


async def run_db_operation(operation: Callable[[], T]) -> T:
    """
    Run a blocking database operation in the thread pool executor.

    This prevents the operation from blocking the asyncio event loop,
    allowing other async tasks to continue while waiting for the database.

    Usage:
        # Instead of:
        result = supabase.table("documents").select("*").execute()

        # Use:
        result = await run_db_operation(
            lambda: supabase.table("documents").select("*").execute()
        )

    Args:
        operation: A callable that performs the database operation

    Returns:
        The result of the database operation
    """
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(_db_executor, operation)
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        raise


async def db_select(supabase, table: str, columns: str = "*", **filters) -> Any:
    """
    Async wrapper for SELECT queries.

    Args:
        supabase: Supabase client instance
        table: Table name
        columns: Columns to select (default "*")
        **filters: Filter conditions as keyword arguments

    Returns:
        Query result
    """
    def _execute():
        query = supabase.table(table).select(columns)
        for key, value in filters.items():
            query = query.eq(key, value)
        return query.execute()

    return await run_db_operation(_execute)


async def db_select_single(supabase, table: str, columns: str = "*", **filters) -> Any:
    """
    Async wrapper for SELECT queries expecting a single result.
    """
    def _execute():
        query = supabase.table(table).select(columns)
        for key, value in filters.items():
            query = query.eq(key, value)
        return query.single().execute()

    return await run_db_operation(_execute)


async def db_insert(supabase, table: str, data: dict | list) -> Any:
    """
    Async wrapper for INSERT queries.

    Args:
        supabase: Supabase client instance
        table: Table name
        data: Data to insert (dict for single row, list for multiple)

    Returns:
        Query result
    """
    def _execute():
        return supabase.table(table).insert(data).execute()

    return await run_db_operation(_execute)


async def db_update(supabase, table: str, data: dict, **filters) -> Any:
    """
    Async wrapper for UPDATE queries.

    Args:
        supabase: Supabase client instance
        table: Table name
        data: Data to update
        **filters: Filter conditions

    Returns:
        Query result
    """
    def _execute():
        query = supabase.table(table).update(data)
        for key, value in filters.items():
            query = query.eq(key, value)
        return query.execute()

    return await run_db_operation(_execute)


async def db_delete(supabase, table: str, **filters) -> Any:
    """
    Async wrapper for DELETE queries.
    """
    def _execute():
        query = supabase.table(table).delete()
        for key, value in filters.items():
            query = query.eq(key, value)
        return query.execute()

    return await run_db_operation(_execute)


async def db_storage_download(supabase, bucket: str, path: str) -> bytes:
    """
    Async wrapper for storage downloads.
    """
    def _execute():
        return supabase.storage.from_(bucket).download(path)

    return await run_db_operation(_execute)


def shutdown_db_executor():
    """
    Shutdown the database executor gracefully.
    Call this when the application is shutting down.
    """
    _db_executor.shutdown(wait=True)
