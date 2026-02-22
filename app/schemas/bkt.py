from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Legacy schemas (kept for existing /update, /update-batch, /mastery, /weak-skills)
# ---------------------------------------------------------------------------

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
    document_id: str
    claude_score: float = Field(..., ge=0, le=100)


class BKTUpdateResponse(BaseModel):
    user_id: str
    question_id: str
    q: float
    updated: list


class BKTBatchItem(BaseModel):
    question_id: str
    claude_score: float = Field(..., ge=0, le=100)


class BKTBatchUpdateRequest(BaseModel):
    user_id: str
    document_id: str
    updates: List[BKTBatchItem]


class BKTBatchUpdateResponse(BaseModel):
    user_id: str
    document_id: str
    count: int
    results: list


# ---------------------------------------------------------------------------
# New adaptive learning schemas
# ---------------------------------------------------------------------------

# --- Session ---

class SessionQuestionOption(BaseModel):
    id: str
    text: str
    index: int


class SessionQuestion(BaseModel):
    id: str
    question: str
    hint: Optional[str] = None
    difficulty_level: str = "medium"
    concept_id: Optional[str] = None
    concept_name: str = "Unknown"
    options: List[SessionQuestionOption]


class SessionConceptSummary(BaseModel):
    concept_id: str
    concept_name: str
    p_mastery: float
    n_attempts: int


class BKTSessionResponse(BaseModel):
    session_id: str
    questions: List[SessionQuestion]
    concepts: List[SessionConceptSummary]
    total_questions: int = 0
    all_mastered: bool = False


# --- Answer ---

class BKTAnswerRequest(BaseModel):
    user_id: str
    question_id: str
    document_id: str
    selected_option_index: int
    session_id: Optional[str] = None
    time_spent_ms: Optional[int] = None


class BKTAnswerResponse(BaseModel):
    question_id: str
    concept_id: Optional[str] = None
    is_correct: bool
    correct_option_index: int
    correct_option_text: str
    explanation: str = ""
    selected_option_index: int
    p_mastery_before: float
    p_mastery_after: float
    is_newly_mastered: bool = False
    mastery_threshold: float


# --- Progress ---

class ConceptProgress(BaseModel):
    concept_id: str
    concept_name: str
    explanation: str = ""
    p_mastery: float
    n_attempts: int = 0
    n_correct: int = 0
    status: str  # "not_started" | "in_progress" | "mastered"
    is_mastered: bool = False
    question_count: int = 0


class TopicProgress(BaseModel):
    topic_id: str
    topic_name: str
    status: str  # "not_started" | "in_progress" | "mastered"
    concepts: List[ConceptProgress]
    total_concepts: int
    mastered_concepts: int
    overall_progress: int  # 0-100


class BKTProgressResponse(BaseModel):
    document_id: str
    document_title: str
    topics: List[TopicProgress]
    total_concepts: int
    mastered_concepts: int
    overall_progress: int  # 0-100
    mastery_threshold: float
