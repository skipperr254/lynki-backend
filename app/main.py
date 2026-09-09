import asyncio
import contextlib
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.auth import get_current_user_id
from app.core.config import get_settings
from app.api.v1.router import api_router
from app.services.watchdog_service import watchdog_loop

settings = get_settings()

if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        integrations=[StarletteIntegration(), FastApiIntegration()],
        traces_sample_rate=0.2,
        environment="production",
    )


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(watchdog_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Router-level auth: nothing under API_V1_STR is reachable without a valid
# Supabase bearer token, including any route added here in the future.
app.include_router(
    api_router,
    prefix=settings.API_V1_STR,
    dependencies=[Depends(get_current_user_id)],
)


# Unauthenticated on purpose — the Render keep-alive pinger and the
# frontend's cold-start wake-up hit these before a user is signed in.
@app.get("/")
def root():
    return {"message": "Welcome to Lynki Backend API"}


@app.get("/health")
def health():
    return {"status": "ok"}
