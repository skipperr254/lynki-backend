"""
Tests for the start_quiz_attempt status guard: a quiz that is still
'generating' or ended up 'failed' must never silently produce a 0-question
attempt (the previous behavior when question_order wasn't populated yet).
"""
from unittest.mock import AsyncMock

import pytest

import app.services.quiz_attempts_service as svc


def _course_quizzes_response(status, error_message=None):
    return {"id": "quiz-1", "name": "Test Quiz", "status": status, "error_message": error_message}


@pytest.fixture(autouse=True)
def stub_db(monkeypatch):
    """run_db_operation returns canned responses keyed by call order isn't needed —
    we only reach the quiz_resp lookup before raising, so a single stub suffices."""
    async def _run(fn):
        class _Resp:
            pass
        return _Resp()
    monkeypatch.setattr(svc, "run_db_operation", _run)


async def _run_start_attempt_with_quiz(monkeypatch, quiz_row):
    async def fake_run(fn):
        class _Resp:
            data = quiz_row
        return _Resp()

    # First call in start_quiz_attempt is the course_quizzes lookup.
    monkeypatch.setattr(svc, "run_db_operation", fake_run)
    return await svc.start_quiz_attempt(user_id="u1", quiz_id="quiz-1", course_id="c1")


@pytest.mark.asyncio
async def test_generating_quiz_is_rejected_not_silently_started(monkeypatch):
    with pytest.raises(ValueError, match="still generating"):
        await _run_start_attempt_with_quiz(monkeypatch, _course_quizzes_response("generating"))


@pytest.mark.asyncio
async def test_failed_quiz_is_rejected_with_its_error_message(monkeypatch):
    with pytest.raises(ValueError, match="Sonnet was unavailable"):
        await _run_start_attempt_with_quiz(
            monkeypatch, _course_quizzes_response("failed", "Sonnet was unavailable")
        )


@pytest.mark.asyncio
async def test_missing_quiz_raises_not_found(monkeypatch):
    async def fake_run(fn):
        class _Resp:
            data = None
        return _Resp()

    monkeypatch.setattr(svc, "run_db_operation", fake_run)
    with pytest.raises(ValueError, match="not found"):
        await svc.start_quiz_attempt(user_id="u1", quiz_id="missing", course_id="c1")
