from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str  # Critical for bypassing RLS during processing
    ANTHROPIC_API_KEY: str
    SENTRY_DSN: str = ""

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Lynki Backend"

    ALLOWED_ORIGINS: str = "https://app.passai.study,https://passai.study"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

@lru_cache()
def get_settings():
    return Settings()  # type: ignore[call-arg]
