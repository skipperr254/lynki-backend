from pydantic import BaseModel


class StudyPlanGenerateRequest(BaseModel):
    user_id: str
    course_id: str


class StudyPlanGenerateResponse(BaseModel):
    plan_json: dict
    generated_at: str  # ISO-8601 string
