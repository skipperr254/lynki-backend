from fastapi import APIRouter
from app.api.v1.endpoints import documents
from app.api.v1.endpoints import bkt
from app.api.v1.endpoints import test
from app.api.v1.endpoints import study_plan
from app.api.v1.endpoints import topic_quiz
from app.api.v1.endpoints import quiz_sessions
from app.api.v1.endpoints import quiz_attempts
from app.api.v1.endpoints import topic_tending

api_router = APIRouter()
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(bkt.router, prefix="/bkt", tags=["bkt"])
api_router.include_router(test.router, prefix="/test", tags=["test"])
api_router.include_router(study_plan.router, prefix="/study-plan", tags=["study-plan"])
api_router.include_router(topic_quiz.router, prefix="/topic-quiz", tags=["topic-quiz"])
api_router.include_router(quiz_sessions.router, prefix="/quiz-sessions", tags=["quiz-sessions"])
api_router.include_router(quiz_attempts.router, prefix="/quiz-attempts", tags=["quiz-attempts"])
api_router.include_router(topic_tending.router, prefix="/topic-tending", tags=["topic-tending"])


