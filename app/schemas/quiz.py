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
    question_format: Literal["standard", "explanatory"] = "standard"
    post_answer_summary: Optional[str] = None  # Shown after answer, regardless of right/wrong


