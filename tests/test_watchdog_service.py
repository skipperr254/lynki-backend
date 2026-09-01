"""
Tests for the watchdog quiz sweep: stale 'generating' rows with questions are
finalized as completed (progressive-start partials are usable quizzes); only
rows with zero questions become failed. Every per-row update must be guarded
with status='generating' so a live job's terminal write wins the race.
"""
import pytest

import app.services.watchdog_service as svc


class FakeTable:
    def __init__(self, store, name):
        self._store = store
        self._name = name
        self._op = None
        self._payload = None
        self._filters = []

    def select(self, *cols):
        self._op = "select"
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    def eq(self, col, val):
        self._filters.append(("eq", col, val))
        return self

    def lt(self, col, val):
        self._filters.append(("lt", col, val))
        return self

    def in_(self, col, vals):
        self._filters.append(("in", col, vals))
        return self

    def execute(self):
        class _Resp:
            data = None

        resp = _Resp()
        if self._op == "select":
            resp.data = self._store["stale_rows"]
        elif self._op == "update":
            self._store["updates"].append(
                {"table": self._name, "payload": self._payload, "filters": list(self._filters)}
            )
            resp.data = [self._payload]
        return resp


class FakeSupabase:
    def __init__(self, stale_rows):
        self.store = {"stale_rows": stale_rows, "updates": []}

    def table(self, name):
        return FakeTable(self.store, name)


@pytest.fixture(autouse=True)
def fast_run_db_operation(monkeypatch):
    async def _run(fn):
        return fn()

    monkeypatch.setattr(svc, "run_db_operation", _run)


def _install(monkeypatch, stale_rows):
    fake = FakeSupabase(stale_rows)
    monkeypatch.setattr(svc, "get_supabase", lambda: fake)
    return fake


@pytest.mark.asyncio
async def test_stale_quiz_with_no_questions_is_failed(monkeypatch):
    fake = _install(monkeypatch, [
        {"id": "quiz-1", "name": "Generating…", "question_order": []},
    ])

    count = await svc.finalize_stuck_quizzes()

    assert count == 1
    (update,) = fake.store["updates"]
    assert update["payload"]["status"] == "failed"
    assert update["payload"]["error_message"]
    assert ("eq", "status", "generating") in update["filters"]


@pytest.mark.asyncio
async def test_stale_quiz_with_partial_questions_is_completed(monkeypatch):
    fake = _install(monkeypatch, [
        {"id": "quiz-1", "name": "Generating…", "question_order": ["q1", "q2", "q3"]},
    ])

    count = await svc.finalize_stuck_quizzes()

    assert count == 1
    (update,) = fake.store["updates"]
    assert update["payload"]["status"] == "completed"
    assert update["payload"]["total_questions"] == 3
    # placeholder name replaced when the job died before naming
    assert update["payload"]["name"] == "Quiz"
    assert ("eq", "status", "generating") in update["filters"]
    assert ("eq", "id", "quiz-1") in update["filters"]


@pytest.mark.asyncio
async def test_stale_quiz_with_real_name_keeps_it(monkeypatch):
    fake = _install(monkeypatch, [
        {"id": "quiz-1", "name": "Cellular Energy", "question_order": ["q1"]},
    ])

    await svc.finalize_stuck_quizzes()

    (update,) = fake.store["updates"]
    assert update["payload"]["status"] == "completed"
    assert "name" not in update["payload"]


@pytest.mark.asyncio
async def test_no_stale_rows_no_updates(monkeypatch):
    fake = _install(monkeypatch, [])

    count = await svc.finalize_stuck_quizzes()

    assert count == 0
    assert fake.store["updates"] == []
