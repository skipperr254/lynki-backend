from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.services.topic_quiz_service import TopicQuizService

router = APIRouter()


class AnswerRequest(BaseModel):
    session_id: str
    question_index: int
    selected_option: int


class CompleteRequest(BaseModel):
    session_id: str


@router.get("/session/{user_id}/{course_id}/{topic_id}")
async def get_or_create_session(user_id: str, course_id: str, topic_id: str):
    """
    Return an active (in_progress) topic quiz session if one exists.
    Otherwise generate a fresh quiz for the topic and return the new session.
    Questions are stripped of is_correct before being sent to the client.
    """
    try:
        return await TopicQuizService.get_or_create_session(user_id, course_id, topic_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer")
async def submit_answer(req: AnswerRequest, background_tasks: BackgroundTasks):
    """
    Returns correctness + explanations immediately (1 DB read).
    Session write and BKT mastery update run in the background after the response.
    """
    try:
        result, session, is_correct = await TopicQuizService.check_answer(
            session_id=req.session_id,
            question_index=req.question_index,
            selected_option=req.selected_option,
        )
        background_tasks.add_task(
            TopicQuizService.persist_answer,
            session,
            req.question_index,
            req.selected_option,
            is_correct,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complete")
async def complete_session(req: CompleteRequest):
    """Mark a topic quiz session as completed."""
    try:
        await TopicQuizService.complete_session(req.session_id)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
