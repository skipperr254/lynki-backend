from __future__ import annotations

from pydantic import BaseModel, Field


class BKTUpdateRequest(BaseModel):
    user_id: str
    question_id: str
    claude_score: float = Field(..., ge=0, le=100)


class BKTUpdateResponse(BaseModel):
    user_id: str
    question_id: str
    q: float
    updated: list
