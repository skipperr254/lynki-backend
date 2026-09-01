"""
Tests for the on-demand quiz generation job: bounded wall-clock time,
partial-success handling, and terminal-status guarantees.

`_generate_and_save_question` (the DB + Claude call) is monkeypatched so
these tests exercise only the orchestration logic in
`run_quiz_generation_job` / `start_quiz_generation` — no live API or DB calls.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

import app.services.on_demand_quiz_service as svc


class FakeTable:
    """Minimal fake of the chain `_supabase.table(...).insert(...).eq(...).execute()`."""

    def __init__(self, recorder, name):
        self._recorder = recorder
        self._name = name
        self._op = None
        self._payload = None

    def insert(self, payload):
        self._op, self._payload = "insert", payload
        return self

    def update(self, payload):
        self._op, self._payload = "update", payload
        return self

    def eq(self, *args, **kwargs):
        return self

    def execute(self):
        self._recorder.append((self._name, self._op, self._payload))

        class _Resp:
            data = [self._payload] if self._payload is not None else []

        return _Resp()


class FakeSupabase:
    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []

    def table(self, name):
        return FakeTable(self.calls, name)

    def last_update(self, table_name):
        updates = [c for c in self.calls if c[0] == table_name and c[1] == "update"]
        assert updates, f"no update recorded for {table_name}"
        return updates[-1][2]


@pytest.fixture(autouse=True)
def fake_supabase(monkeypatch):
    fake = FakeSupabase()
    monkeypatch.setattr(svc, "_supabase", fake)
    return fake


@pytest.fixture(autouse=True)
def fast_run_db_operation(monkeypatch):
    """run_db_operation just awaits the sync callable's .execute() result directly."""
    async def _run(fn):
        return fn()
    monkeypatch.setattr(svc, "run_db_operation", _run)


@pytest.fixture(autouse=True)
def no_op_side_calls(monkeypatch):
    monkeypatch.setattr(svc, "_get_mastery_map", AsyncMock(return_value={}))
    monkeypatch.setattr(svc, "_get_prior_question_stems", AsyncMock(return_value=[]))
    monkeypatch.setattr(svc, "_generate_quiz_name", AsyncMock(return_value="Test Quiz"))


def _concepts(n):
    return [{"id": f"concept-{i}", "name": f"Concept {i}"} for i in range(n)]


@pytest.mark.asyncio
async def test_all_questions_succeed_marks_completed(fake_supabase, monkeypatch):
    async def fake_generate(concept, quiz_id, semaphore, excluded_questions=None):
        return f"question-{concept['id']}"

    monkeypatch.setattr(svc, "_generate_and_save_question", fake_generate)

    await svc.run_quiz_generation_job(
        quiz_id="quiz-1", user_id="u1", course_id="c1", concepts=_concepts(3), quiz_size=3
    )

    update = fake_supabase.last_update("course_quizzes")
    assert update["status"] == "completed"
    assert update["total_questions"] == 3
    assert set(update["question_order"]) == {"question-concept-0", "question-concept-1", "question-concept-2"}


@pytest.mark.asyncio
async def test_partial_success_still_marks_completed_with_fewer_questions(fake_supabase, monkeypatch):
    async def fake_generate(concept, quiz_id, semaphore, excluded_questions=None):
        if concept["id"] == "concept-1":
            return None  # this one "failed" to generate, like a persistent validation error
        return f"question-{concept['id']}"

    monkeypatch.setattr(svc, "_generate_and_save_question", fake_generate)

    await svc.run_quiz_generation_job(
        quiz_id="quiz-1", user_id="u1", course_id="c1", concepts=_concepts(3), quiz_size=3
    )

    update = fake_supabase.last_update("course_quizzes")
    assert update["status"] == "completed"
    assert update["total_questions"] == 2


@pytest.mark.asyncio
async def test_zero_questions_marks_failed_with_message(fake_supabase, monkeypatch):
    async def fake_generate(concept, quiz_id, semaphore, excluded_questions=None):
        return None

    monkeypatch.setattr(svc, "_generate_and_save_question", fake_generate)

    await svc.run_quiz_generation_job(
        quiz_id="quiz-1", user_id="u1", course_id="c1", concepts=_concepts(2), quiz_size=2
    )

    update = fake_supabase.last_update("course_quizzes")
    assert update["status"] == "failed"
    assert update["error_message"]


@pytest.mark.asyncio
async def test_job_never_exceeds_its_time_budget_and_keeps_partial_results(fake_supabase, monkeypatch):
    monkeypatch.setattr(svc, "QUIZ_GENERATION_TIMEOUT", 0.1)

    async def fast(concept, quiz_id, semaphore, excluded_questions=None):
        return f"question-{concept['id']}"

    async def slow(concept, quiz_id, semaphore, excluded_questions=None):
        await asyncio.sleep(10)
        return f"question-{concept['id']}"  # pragma: no cover — should be cancelled first

    async def fake_generate(concept, quiz_id, semaphore, excluded_questions=None):
        if concept["id"] == "concept-slow":
            return await slow(concept, quiz_id, semaphore, excluded_questions)
        return await fast(concept, quiz_id, semaphore, excluded_questions)

    monkeypatch.setattr(svc, "_generate_and_save_question", fake_generate)

    concepts = _concepts(2) + [{"id": "concept-slow", "name": "Slow concept"}]

    elapsed = asyncio.get_event_loop().time()
    await svc.run_quiz_generation_job(
        quiz_id="quiz-1", user_id="u1", course_id="c1", concepts=concepts, quiz_size=3
    )
    elapsed = asyncio.get_event_loop().time() - elapsed

    assert elapsed < 5  # bounded by QUIZ_GENERATION_TIMEOUT, not the 10s sleep
    update = fake_supabase.last_update("course_quizzes")
    assert update["status"] == "completed"
    assert update["total_questions"] == 2  # the slow one was cancelled, not counted


@pytest.mark.asyncio
async def test_question_order_is_published_once_in_the_terminal_write(
    fake_supabase, monkeypatch
):
    """course_quizzes is written exactly once by the job — the terminal update.

    The job used to flush question_order after every completed batch so the
    frontend could start the quiz before generation finished. Measured, the
    3rd of 10 concurrent questions landed ~2s before the last and the poll
    interval consumed all of it, so the partial writes are gone. A regression
    would show up here as an extra update carrying no status.
    """
    delays = {"concept-0": 0.01, "concept-1": 0.03, "concept-2": 0.05}

    async def fake_generate(concept, quiz_id, semaphore, excluded_questions=None):
        await asyncio.sleep(delays[concept["id"]])
        return f"question-{concept['id']}"

    monkeypatch.setattr(svc, "_generate_and_save_question", fake_generate)

    await svc.run_quiz_generation_job(
        quiz_id="quiz-1", user_id="u1", course_id="c1", concepts=_concepts(3), quiz_size=3
    )

    updates = [c[2] for c in fake_supabase.calls if c[0] == "course_quizzes" and c[1] == "update"]
    assert len(updates) == 1
    assert updates[0]["status"] == "completed"
    assert len(updates[0]["question_order"]) == 3


@pytest.mark.asyncio
async def test_question_order_follows_selection_not_completion(fake_supabase, monkeypatch):
    """No shuffle, and no dependence on which task happens to finish first.

    Order comes from the concept-selection order (the job iterates its task
    list, not the `done` set, precisely so this is deterministic), even when
    the concepts complete in reverse.
    """
    delays = {"concept-0": 0.05, "concept-1": 0.03, "concept-2": 0.01}

    async def fake_generate(concept, quiz_id, semaphore, excluded_questions=None):
        await asyncio.sleep(delays[concept["id"]])
        return f"question-{concept['id']}"

    monkeypatch.setattr(svc, "_generate_and_save_question", fake_generate)
    # Selection is weighted-random; pin it so the expected order is fixed.
    monkeypatch.setattr(
        svc, "_select_concepts", lambda concepts, mastery_map, quiz_size: concepts[:quiz_size]
    )

    await svc.run_quiz_generation_job(
        quiz_id="quiz-1", user_id="u1", course_id="c1", concepts=_concepts(3), quiz_size=3
    )

    update = fake_supabase.last_update("course_quizzes")
    assert update["question_order"] == [
        "question-concept-0", "question-concept-1", "question-concept-2"
    ]


class FailOnceOnCompletedSupabase(FakeSupabase):
    """Raises on the first course_quizzes update carrying status='completed'
    (the job's terminal write), then behaves normally — so the outer except
    handler's finalize write succeeds."""

    def __init__(self):
        super().__init__()
        self.failed_once = False

    def table(self, name):
        outer = self

        class _Table(FakeTable):
            def execute(self):
                if (
                    not outer.failed_once
                    and self._op == "update"
                    and isinstance(self._payload, dict)
                    and self._payload.get("status") == "completed"
                ):
                    outer.failed_once = True
                    raise RuntimeError("terminal write failed")
                return super().execute()

        return _Table(self.calls, name)


@pytest.mark.asyncio
async def test_crash_after_partial_questions_finalizes_completed(monkeypatch):
    """Invariant: 'failed' ⇔ zero questions. A job that crashes after saving
    questions must finalize the quiz as completed with what exists."""
    fake = FailOnceOnCompletedSupabase()
    monkeypatch.setattr(svc, "_supabase", fake)

    async def fake_generate(concept, quiz_id, semaphore, excluded_questions=None):
        return f"question-{concept['id']}"

    monkeypatch.setattr(svc, "_generate_and_save_question", fake_generate)

    await svc.run_quiz_generation_job(
        quiz_id="quiz-1", user_id="u1", course_id="c1", concepts=_concepts(3), quiz_size=3
    )

    update = fake.last_update("course_quizzes")
    assert update["status"] == "completed"
    assert update["total_questions"] == 3
    assert set(update["question_order"]) == {
        "question-concept-0", "question-concept-1", "question-concept-2"
    }


@pytest.mark.asyncio
async def test_start_quiz_generation_returns_immediately_without_generating(fake_supabase, monkeypatch):
    monkeypatch.setattr(svc, "_get_concepts_for_course", AsyncMock(return_value=_concepts(2)))
    generate_mock = AsyncMock()
    monkeypatch.setattr(svc, "_generate_and_save_question", generate_mock)

    result = await svc.start_quiz_generation(user_id="u1", course_id="c1", quiz_size=2)

    assert result["status"] == "generating"
    assert result["quiz_id"] is not None
    assert "concepts" in result
    generate_mock.assert_not_called()

    insert = [c for c in fake_supabase.calls if c[1] == "insert"][0]
    assert insert[2]["status"] == "generating"


@pytest.mark.asyncio
async def test_start_quiz_generation_fails_fast_with_no_concepts(fake_supabase, monkeypatch):
    monkeypatch.setattr(svc, "_get_concepts_for_course", AsyncMock(return_value=[]))

    result = await svc.start_quiz_generation(user_id="u1", course_id="c1", quiz_size=2)

    assert result["quiz_id"] is None
    assert result["error"]
    assert fake_supabase.calls == []  # no placeholder row created
