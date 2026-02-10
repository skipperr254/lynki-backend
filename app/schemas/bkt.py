from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List


class BKTSkill(BaseModel):
    skill_name: str
    mastery: float
    attempts: int


class BKTSummaryResponse(BaseModel):
    pass_probability: float
    skills: List[BKTSkill]


class BKTWeakSkillsResponse(BaseModel):
    skills: List[BKTSkill]

class BKTUpdateRequest(BaseModel):
    user_id: str
    question_id: str
    claude_score: float = Field(..., ge=0, le=100)


class BKTUpdateResponse(BaseModel):
    user_id: str
    question_id: str
    q: float
    updated: list
