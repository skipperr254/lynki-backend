from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str  # Critical for bypassing RLS during processing
    ANTHROPIC_API_KEY: str
    
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Lynki Backend"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

@lru_cache()
def get_settings():
    return Settings()  # type: ignore[call-arg]
