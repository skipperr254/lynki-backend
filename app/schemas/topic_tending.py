"""
Pydantic schemas for the /topic-tending/* endpoints.

These match the TypeScript types in the frontend (src/features/tending/types.ts).
"""

from __future__ import annotations

from typing import List, Optional, Any, Dict
from pydantic import BaseModel


# ─── Sub-models mirroring the frontend types ──────────────────────────────────

class RecallCardResult(BaseModel):
    id: str
    rating: str  # "got_it" | "again"


class ActiveRecallResult(BaseModel):
    student_response: str
    evaluation: Optional[Dict[str, Any]] = None  # RecallEvaluation | null


class MnemonicResult(BaseModel):
    id: str
    lockedIn: bool


class ConnectionResult(BaseModel):
    id: str
    attempts: int
    matched: bool


class QuizResult(BaseModel):
    correct: int
    total: int
    question_ids: List[str]


class AllStageResults(BaseModel):
    recall: Optional[List[RecallCardResult]] = None
    active_recall: Optional[ActiveRecallResult] = None
    mnemonics: Optional[List[MnemonicResult]] = None
    connections: Optional[List[ConnectionResult]] = None
    quiz: Optional[QuizResult] = None
    stages_skipped: List[str] = []


# ─── Generate endpoint ────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    user_id: str
    course_id: str
    topic_id: str


class RecallCard(BaseModel):
    id: str
    front: str
    back: str


class MnemonicCard(BaseModel):
    id: str
    hook: str
    explanation: str


class ConnectionPair(BaseModel):
    id: str
    left: str
    right: str


class GenerateResponsePayload(BaseModel):
    """
    The generated_content JSON stored in the DB and returned to the frontend.
    Matches TendingSessionPayload in types.ts.
    """
    session_id: str
    course_id: str
    topic_id: str
    topic_title: str
    recall_cards: Dict[str, Any]   # { stage: "recall_cards", cards: [...] }
    active_recall: Dict[str, Any]  # { stage: "active_recall", prompt: "...", source_paragraph: "..." }
    mnemonics: Dict[str, Any]      # { stage: "mnemonics", mnemonics: [...] }
    connections: Dict[str, Any]    # { stage: "connections", pairs: [...], type: "..." }


# ─── Evaluate-recall endpoint ─────────────────────────────────────────────────

class EvaluateRecallRequest(BaseModel):
    session_id: str
    student_response: str


class EvaluateRecallResponse(BaseModel):
    got_right: List[str]
    missed: List[str]
    source_paragraph: str


# ─── Complete endpoint ────────────────────────────────────────────────────────

class CompleteRequest(BaseModel):
    session_id: str
    results: AllStageResults


class KCDelta(BaseModel):
    kc_id: str
    name: str
    before: float
    after: float


class CompleteResponse(BaseModel):
    """
    Matches MasteryDelta in the frontend types.ts — this is what the Mastery
    Delta screen renders.
    """
    stage: str = "mastery_delta"
    topic_title: str
    mastery_before: float
    mastery_after: float
    kc_breakdown: List[KCDelta]
    tended_today: bool = True
