"""
Tests for the processing_stage sub-status writes added to ExtractionService.
The frontend renders these to turn a static "Processing" spinner into a
staged, narrated wait — these tests just verify the correct rows/values get
written, not the full extraction pipeline (already covered elsewhere).
"""
import pytest

from app.services.extraction_service import ExtractionService


class FakeTable:
    def __init__(self, store, name):
        self._store = store
        self._name = name
        self._payload = None
        self._filters = []

    def update(self, payload):
        self._payload = payload
        return self

    def eq(self, col, val):
        self._filters.append((col, val))
        return self

    def execute(self):
        self._store.append(
            {"table": self._name, "payload": self._payload, "filters": list(self._filters)}
        )

        class _Resp:
            data = [self._payload]

        return _Resp()


class FakeSupabase:
    def __init__(self):
        self.updates = []

    def table(self, name):
        return FakeTable(self.updates, name)


def _new_service() -> ExtractionService:
    """Skip __init__ (constructs a real AsyncAnthropic client) — these methods
    only need self.supabase, which is stubbed below."""
    svc = ExtractionService.__new__(ExtractionService)
    svc.supabase = FakeSupabase()
    return svc


@pytest.fixture(autouse=True)
def fast_run_db_operation(monkeypatch):
    async def _run(fn):
        return fn()

    monkeypatch.setattr("app.services.extraction_service.run_db_operation", _run)


@pytest.mark.asyncio
async def test_start_processing_sets_status_stage_and_started_at():
    svc = _new_service()

    await svc._start_processing("doc-1")

    assert len(svc.supabase.updates) == 1
    update = svc.supabase.updates[0]
    assert update["table"] == "documents"
    assert update["filters"] == [("id", "doc-1")]
    assert update["payload"]["status"] == "processing"
    assert update["payload"]["processing_stage"] == "extracting"
    assert update["payload"]["processing_started_at"]  # non-empty ISO timestamp


@pytest.mark.asyncio
async def test_update_processing_stage_writes_only_the_stage():
    svc = _new_service()

    await svc._update_processing_stage("doc-1", "analyzing")

    assert len(svc.supabase.updates) == 1
    update = svc.supabase.updates[0]
    assert update["payload"] == {"processing_stage": "analyzing"}
    assert update["filters"] == [("id", "doc-1")]


@pytest.mark.asyncio
async def test_start_processing_then_analyzing_transitions_in_order():
    svc = _new_service()

    await svc._start_processing("doc-1")
    await svc._update_processing_stage("doc-1", "analyzing")

    stages = [u["payload"].get("processing_stage") for u in svc.supabase.updates]
    assert stages == ["extracting", "analyzing"]
