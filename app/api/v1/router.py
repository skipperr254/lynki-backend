from fastapi import APIRouter
from app.api.v1.endpoints import documents, quizzes
from app.api.v1.endpoints import bkt
from app.api.v1.endpoints import test

api_router = APIRouter()
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(quizzes.router, prefix="/quizzes", tags=["quizzes"])
api_router.include_router(bkt.router, prefix="/bkt", tags=["bkt"])
api_router.include_router(test.router, prefix="/test", tags=["test"])


