from fastapi import APIRouter
from app.api.v1.endpoints import documents, quizzes

api_router = APIRouter()
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(quizzes.router, prefix="/quizzes", tags=["quizzes"])
