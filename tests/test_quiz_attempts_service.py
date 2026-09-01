"""
Tests for the start_quiz_attempt status guard: a quiz that is still
'generating' with no questions yet, or one that ended up 'failed', must never
silently produce a 0-question attempt — while a 'generating' quiz that already
has questions (progressive start) starts normally with the partial list.
"""
from unittest.mock import AsyncMock

import pytest

import app.services.quiz_attempts_service as svc


def _course_quizzes_response(status, error_message=None, question_order=None):
    return {
        "id": "quiz-1",
        "name": "Test Quiz",
        "status": status,
        "error_message": error_message,
        "question_order": question_order or [],
    }


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


def _sequenced_db(monkeypatch, responses):
    """run_db_operation pops one canned response per call, in order."""
    queue = list(responses)

    async def fake_run(fn):
        class _Resp:
            data = queue.pop(0)
        return _Resp()

    monkeypatch.setattr(svc, "run_db_operation", fake_run)


def _question_row(qid, concept_id="con-1"):
    return {
        "id": qid,
        "question": f"What is {qid}?",
        "hint": None,
        "difficulty_level": "medium",
        "concept_id": concept_id,
        "question_options": [
            {
                "id": f"{qid}-opt-{i}",
                "option_text": f"Option {i}",
                "option_index": i,
                "is_correct": i == 0,
                "explanation": "Because.",
            }
            for i in range(4)
        ],
    }


@pytest.mark.asyncio
async def test_generating_quiz_with_no_questions_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="still generating"):
        await _run_start_attempt_with_quiz(monkeypatch, _course_quizzes_response("generating"))


@pytest.mark.asyncio
async def test_generating_quiz_with_partial_questions_starts(monkeypatch):
    """Progressive start: 'generating' + non-empty question_order proceeds and
    returns whatever questions exist so far."""
    partial_order = ["q1", "q2", "q3"]
    _sequenced_db(monkeypatch, [
        # 1. course_quizzes lookup (the gate)
        _course_quizzes_response("generating", question_order=partial_order),
        # 2. courses title lookup
        {"title": "Course 1"},
        # 3. quiz_attempts insert
        [{"id": "attempt-1"}],
        # 4. _fetch_quiz_questions_ordered: question_order re-read
        {"question_order": partial_order},
        # 5. questions select
        [_question_row(qid) for qid in partial_order],
        # 6. _get_concept_names: concepts select
        [{"id": "con-1", "name": "Concept One"}],
    ])

    result = await svc.start_quiz_attempt(user_id="u1", quiz_id="quiz-1", course_id="c1")

    assert result["total_questions"] == 3
    assert [q["id"] for q in result["questions"]] == partial_order
    assert result["questions"][0]["concept_name"] == "Concept One"


@pytest.mark.asyncio
async def test_failed_quiz_is_rejected_with_its_error_message(monkeypatch):
    with pytest.raises(ValueError, match="Sonnet was unavailable"):
        await _run_start_attempt_with_quiz(
            monkeypatch, _course_quizzes_response("failed", "Sonnet was unavailable")
        )


@pytest.mark.asyncio
async def test_failed_quiz_is_rejected_even_with_questions(monkeypatch):
    """'failed' now implies zero usable questions, but the gate must reject
    regardless of what question_order holds."""
    with pytest.raises(ValueError, match="generate a new quiz"):
        await _run_start_attempt_with_quiz(
            monkeypatch, _course_quizzes_response("failed", None, question_order=["q1"])
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
