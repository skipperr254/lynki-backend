"""
Study Plan endpoint — generates AI-powered study plans via Claude Haiku.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.study_plan import StudyPlanGenerateRequest, StudyPlanGenerateResponse
from app.services.study_plan_service import generate_study_plan

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/generate", response_model=StudyPlanGenerateResponse)
async def generate_study_plan_endpoint(request: StudyPlanGenerateRequest):
    """
    Generate (or regenerate) an AI-powered study plan for a user+course pair.
    Fetches BKT mastery data internally, calls Claude Haiku, upserts the
    result into the study_plans table, and returns plan_text + generated_at.
    """
    try:
        result = await generate_study_plan(
            user_id=request.user_id,
            course_id=request.course_id,
        )
        return StudyPlanGenerateResponse(
            plan_json=result["plan_json"],
            generated_at=result["generated_at"],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            "Study plan generation failed user=%s course=%s: %s",
            request.user_id,
            request.course_id,
            e,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate study plan. Please try again.",
        )
