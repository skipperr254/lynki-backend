from supabase import create_client, Client
from app.core.config import get_settings

settings = get_settings()

def get_supabase() -> Client:
    # Use Service Role Key to bypass RLS when acting as the system processing files
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
