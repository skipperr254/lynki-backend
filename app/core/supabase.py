from functools import lru_cache

from supabase import create_client, Client
from app.core.config import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """
    Shared Supabase client, built with the Service Role Key so server-side
    work bypasses RLS.

    Cached deliberately. `create_client` constructs a fresh postgrest/storage/
    auth stack with its own connection pool, so calling it per request meant
    every query paid a new TLS handshake to eu-west-2. The client is
    thread-safe (httpx underneath), which matters because `run_db_operation`
    hands queries to a ThreadPoolExecutor.
    """
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
