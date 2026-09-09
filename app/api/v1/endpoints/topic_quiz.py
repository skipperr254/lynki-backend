from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from app.core.async_db import db_select_single
from app.core.auth import get_current_user_id
from app.core.supabase import get_supabase
from app.services.topic_quiz_service import TopicQuizService

router = APIRouter()
_supabase = get_supabase()


async def _assert_session_owner(session_id: str, caller: str) -> None:
    """
    topic_quiz_sessions carries no user_id in the request body — only
    session_id — so ownership has to be checked by loading the row.
    """
    resp = await db_select_single(
        _supabase, "topic_quiz_sessions", "user_id", id=session_id
    )
    session = getattr(resp, "data", None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["user_id"] != caller:
        raise HTTPException(status_code=403, detail="Not your data")


class AnswerRequest(BaseModel):
    session_id: str
    question_index: int
    selected_option: int


class CompleteRequest(BaseModel):
    session_id: str


@router.get("/session/{user_id}/{course_id}/{topic_id}")
async def get_or_create_session(
    user_id: str,
    course_id: str,
    topic_id: str,
    format: str = "standard",
    caller: str = Depends(get_current_user_id),
):
    """
    Return an active (in_progress) topic quiz session if one exists.
    Otherwise generate a fresh quiz for the topic and return the new session.
    Questions are stripped of is_correct before being sent to the client.
    """
    if user_id != caller:
        raise HTTPException(status_code=403, detail="Not your data")
    try:
        return await TopicQuizService.get_or_create_session(
            user_id, course_id, topic_id, format
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer")
async def submit_answer(
    req: AnswerRequest,
    background_tasks: BackgroundTasks,
    caller: str = Depends(get_current_user_id),
):
    """
    Returns correctness + explanations immediately (1 DB read).
    Session write and BKT mastery update run in the background after the response.
    """
    await _assert_session_owner(req.session_id, caller)
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
async def complete_session(req: CompleteRequest, caller: str = Depends(get_current_user_id)):
    """Mark a topic quiz session as completed."""
    await _assert_session_owner(req.session_id, caller)
    try:
        await TopicQuizService.complete_session(req.session_id)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
