from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# ============================================
# Quiz Generation Schemas
# ============================================

class QuestionOption(BaseModel):
    """Individual answer option with explanation"""
    option_text: str
    option_index: int
    is_correct: bool
    explanation: str  # Why this option is correct or incorrect


class GeneratedQuestion(BaseModel):
    """A single generated quiz question"""
    question: str
    options: List[QuestionOption] = Field(min_length=2, max_length=6)
    hint: Optional[str] = None
    difficulty_level: Literal["easy", "medium", "hard"] = "medium"
    concept_id: str  # Link to knowledge component


class QuizGenerationRequest(BaseModel):
    """Request to generate quiz for a document"""
    document_id: str
    questions_per_concept: int = Field(default=3, ge=1, le=10)
    include_hints: bool = True


class QuizGenerationResponse(BaseModel):
    """Response after quiz generation"""
    quiz_id: str
    document_id: str
    status: Literal["pending", "generating", "completed", "failed"]
    total_questions: int
    message: str


class ConceptWithQuestions(BaseModel):
    """Concept with its generated questions"""
    concept_id: str
    concept_name: str
    questions: List[GeneratedQuestion]


# ============================================
# Quiz Retrieval Schemas
# ============================================

class QuestionOptionResponse(BaseModel):
    """Frontend-friendly question option"""
    id: str
    option_text: str
    option_index: int
    is_correct: bool
    explanation: str


class QuestionResponse(BaseModel):
    """Frontend-friendly question with all details"""
    id: str
    question: str
    options: List[QuestionOptionResponse]
    hint: Optional[str] = None
    difficulty_level: str
    concept_id: Optional[str] = None
    order_index: int


class QuizResponse(BaseModel):
    """Frontend-friendly quiz with questions"""
    id: str
    title: str
    description: str
    document_id: Optional[str] = None
    user_id: Optional[str] = None
    generation_status: str
    questions: List[QuestionResponse]
    created_at: str
    updated_at: str


class QuizListItem(BaseModel):
    """Minimal quiz info for listing"""
    id: str
    title: str
    description: str
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    generation_status: str
    question_count: int
    created_at: str


# ============================================
# Quiz Attempt Schemas
# ============================================

class QuizAnswer(BaseModel):
    """User's answer to a question"""
    question_id: str
    selected_option_index: int


class QuizAttemptSubmit(BaseModel):
    """Submit quiz attempt"""
    quiz_id: str
    answers: List[QuizAnswer]


class QuestionResult(BaseModel):
    """Result for a single question"""
    question_id: str
    question_text: str
    selected_option_index: int
    correct_option_index: int
    is_correct: bool
    explanation: str  # Explanation for the selected answer
    hint: Optional[str] = None


class QuizAttemptResult(BaseModel):
    """Complete quiz attempt results"""
    attempt_id: str
    quiz_id: str
    score: int
    total_questions: int
    percentage: float
    question_results: List[QuestionResult]
    completed_at: str
