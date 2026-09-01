"""
Tests for analysis_service's concurrent chunk analysis and batched save: the
cross-chunk topic dedup, exception-abort semantics, best-effort handling of
individual chunk retry exhaustion, reuse of a reprocessed document's existing
topics, the per-topic fallback when a bulk concept insert is rejected, and the
round-trip count the batching exists to hold down.

`_analyze_chunk` (the Claude call) is monkeypatched so these tests exercise
only the orchestration logic in `analyze_document` — no live API calls.
`AnalysisService.__init__` is bypassed (it constructs a real AsyncAnthropic
client) via `__new__`, matching this repo's existing pattern of avoiding that
~1s construction cost in tests that don't need it (see
test_topic_quiz_service.py's `_fake_generator_class` comment).
"""
import asyncio

import pytest

from app.services.analysis_service import AnalysisService


class FakeUsage:
    input_tokens = 10
    output_tokens = 10


class FakeTopicsTable:
    """Stateful fake of `topics`: inserted rows persist in a dict shared
    across calls, so a SELECT by document_id returns what an earlier insert
    (or an earlier processing run) put there and the reuse path is really
    exercised, rather than a static canned response."""

    def __init__(self, recorder, topics: dict):
        self._recorder = recorder
        self._topics = topics  # (document_id, name) -> id
        self._filters: dict = {}
        self._insert_payload = None

    def select(self, *_args, **_kwargs):
        return self

    def insert(self, payload):
        self._insert_payload = payload
        return self

    def eq(self, key, value):
        self._filters[key] = value
        return self

    def execute(self):
        if self._insert_payload is not None:
            rows = self._insert_payload
            assert isinstance(rows, list), "topics insert should be batched into one call"
            self._recorder.append(("topics", "insert", rows))
            out = []
            for row in rows:
                new_id = f"topic-{len(self._topics)}"
                self._topics[(row["document_id"], row["name"])] = new_id
                # Deliberately reversed: _save_structure must map the
                # returned representation by name, never by position.
                out.insert(0, {"id": new_id, "name": row["name"]})

            class _Resp:
                data = out

            return _Resp()

        document_id = self._filters.get("document_id")
        self._recorder.append(("topics", "select", {"document_id": document_id}))
        rows = [
            {"id": topic_id, "name": name}
            for (doc, name), topic_id in self._topics.items()
            if doc == document_id
        ]

        class _Resp:
            data = rows

        return _Resp()


class FakeConceptsTable:
    def __init__(self, recorder, fail_next: dict):
        self._recorder = recorder
        self._fail_next = fail_next
        self._payload = None

    def insert(self, payload):
        self._payload = payload
        return self

    def execute(self):
        rows = self._payload if isinstance(self._payload, list) else [self._payload]
        if self._fail_next.get("concepts"):
            # Only the first (bulk) insert fails; the per-topic fallback
            # inserts that follow succeed.
            self._fail_next["concepts"] = False
            self._recorder.append(("concepts", "insert_failed", rows))
            raise RuntimeError("bulk insert rejected")

        self._recorder.append(("concepts", "insert", rows))

        class _Resp:
            data = rows

        return _Resp()


class FakeLlmLogsTable:
    def __init__(self, recorder):
        self._recorder = recorder
        self._payload = None

    def insert(self, payload):
        self._payload = payload
        return self

    def execute(self):
        rows = self._payload if isinstance(self._payload, list) else [self._payload]
        self._recorder.append(("llm_logs", "insert", rows))

        class _Resp:
            data = rows

        return _Resp()


class FakeSupabase:
    def __init__(self):
        self.calls: list[tuple[str, str, object]] = []
        self._topics_state: dict = {}
        self.fail_next: dict = {}

    def seed_topic(self, document_id: str, name: str, topic_id: str):
        self._topics_state[(document_id, name)] = topic_id

    def table(self, name):
        if name == "topics":
            return FakeTopicsTable(self.calls, self._topics_state)
        if name == "concepts":
            return FakeConceptsTable(self.calls, self.fail_next)
        if name == "llm_logs":
            return FakeLlmLogsTable(self.calls)
        raise AssertionError(f"unexpected table: {name}")

    # -- convenience accessors ------------------------------------------
    def rows(self, table: str, op: str = "insert") -> list:
        out = []
        for t, o, payload in self.calls:
            if t == table and o == op:
                out.extend(payload)
        return out

    def count(self, table: str, op: str = "insert") -> int:
        return sum(1 for t, o, _ in self.calls if t == table and o == op)


def _make_service(fake_supabase, monkeypatch) -> AnalysisService:
    svc = AnalysisService.__new__(AnalysisService)
    svc.supabase = fake_supabase
    svc.client = None
    svc.model = "claude-sonnet-4-6"

    async def _run(fn):
        # A real yield point (unlike a bare `return fn()`) — matters for
        # test_cross_chunk_topic_saved_once_with_all_concepts: it's what
        # would let two *concurrently scheduled* saves interleave their
        # SELECT/INSERT and race, if a future regression moved saving back
        # inside the concurrent per-chunk tasks. Mirrors the real
        # run_db_operation, which awaits a thread-pool executor.
        await asyncio.sleep(0)
        return fn()

    monkeypatch.setattr("app.services.analysis_service.run_db_operation", _run)
    return svc


def _chunk_data(topic_name: str, concept_name: str) -> dict:
    return {
        "topics": [
            {
                "name": topic_name,
                "concepts": [
                    {"name": concept_name, "explanation": "e", "source_text": "s"}
                ],
            }
        ]
    }


def _fixed_chunks(*chunks):
    return lambda text, chunk_size=8000: list(chunks)


@pytest.mark.asyncio
async def test_cross_chunk_topic_saved_once_with_all_concepts(monkeypatch):
    """Two chunks both name a topic 'Cell Biology'. Exactly one topic row
    should be written, with both chunks' concepts under it, even when the
    chunks resolve out of (chunk) order."""
    fake_supabase = FakeSupabase()
    svc = _make_service(fake_supabase, monkeypatch)

    async def fake_analyze_chunk(document_id, text_chunk, chunk_index, total_chunks, semaphore):
        # Force out-of-order completion: chunk 0 finishes after chunk 1.
        if chunk_index == 0:
            await asyncio.sleep(0.02)
        return _chunk_data("Cell Biology", f"Concept {chunk_index}"), FakeUsage()

    monkeypatch.setattr(svc, "_analyze_chunk", fake_analyze_chunk)
    monkeypatch.setattr(svc, "_chunk_text", _fixed_chunks("chunk a", "chunk b"))

    await svc.analyze_document("doc-1", "x" * 60)

    topic_rows = fake_supabase.rows("topics")
    assert len(topic_rows) == 1
    assert topic_rows[0]["name"] == "Cell Biology"

    concept_rows = fake_supabase.rows("concepts")
    assert len(concept_rows) == 2  # both concepts still saved
    assert {r["name"] for r in concept_rows} == {"Concept 0", "Concept 1"}
    # ...both under the single shared topic, not two separate ones
    assert len({r["topic_id"] for r in concept_rows}) == 1


@pytest.mark.asyncio
async def test_unexpected_exception_aborts_whole_document_nothing_saved(monkeypatch):
    """Saves are deferred until after the whole concurrent API-call phase
    succeeds, so an unexpected error in any chunk means nothing from this
    run is saved — not just chunks after the failing one."""
    fake_supabase = FakeSupabase()
    svc = _make_service(fake_supabase, monkeypatch)

    async def fake_analyze_chunk(document_id, text_chunk, chunk_index, total_chunks, semaphore):
        if chunk_index == 1:
            raise RuntimeError("boom")
        await asyncio.sleep(0.05)  # chunk 0 would "finish" after chunk 1 raises
        return _chunk_data("Topic A", "Concept A"), FakeUsage()

    monkeypatch.setattr(svc, "_analyze_chunk", fake_analyze_chunk)
    monkeypatch.setattr(svc, "_chunk_text", _fixed_chunks("chunk a", "chunk b"))

    with pytest.raises(RuntimeError, match="boom"):
        await svc.analyze_document("doc-1", "x" * 60)

    assert fake_supabase.calls == []


@pytest.mark.asyncio
async def test_chunk_retry_exhaustion_does_not_block_other_chunks(monkeypatch):
    """A chunk that exhausts retries returns None (not an exception) — the
    other chunks should still save normally, and only the surviving chunk's
    token usage should be logged."""
    fake_supabase = FakeSupabase()
    svc = _make_service(fake_supabase, monkeypatch)

    async def fake_analyze_chunk(document_id, text_chunk, chunk_index, total_chunks, semaphore):
        if chunk_index == 0:
            return None  # retries exhausted, best-effort skip
        return _chunk_data("Topic B", "Concept B"), FakeUsage()

    monkeypatch.setattr(svc, "_analyze_chunk", fake_analyze_chunk)
    monkeypatch.setattr(svc, "_chunk_text", _fixed_chunks("chunk a", "chunk b"))

    await svc.analyze_document("doc-1", "x" * 60)

    topic_rows = fake_supabase.rows("topics")
    assert len(topic_rows) == 1
    assert topic_rows[0]["name"] == "Topic B"
    assert len(fake_supabase.rows("concepts")) == 1
    assert len(fake_supabase.rows("llm_logs")) == 1


@pytest.mark.asyncio
async def test_all_chunks_failing_touches_the_database_not_at_all(monkeypatch):
    """Nothing extracted means nothing to write — no empty inserts."""
    fake_supabase = FakeSupabase()
    svc = _make_service(fake_supabase, monkeypatch)

    async def fake_analyze_chunk(document_id, text_chunk, chunk_index, total_chunks, semaphore):
        return None

    monkeypatch.setattr(svc, "_analyze_chunk", fake_analyze_chunk)
    monkeypatch.setattr(svc, "_chunk_text", _fixed_chunks("chunk a", "chunk b"))

    await svc.analyze_document("doc-1", "x" * 60)

    assert fake_supabase.calls == []


@pytest.mark.asyncio
async def test_reprocessed_document_reuses_existing_topic_rows(monkeypatch):
    """A document reprocessed after a partial failure already has topic rows.
    They should be reused, not duplicated — this is what the surviving SELECT
    round trip is for."""
    fake_supabase = FakeSupabase()
    fake_supabase.seed_topic("doc-1", "Topic A", "topic-existing")
    svc = _make_service(fake_supabase, monkeypatch)

    async def fake_analyze_chunk(document_id, text_chunk, chunk_index, total_chunks, semaphore):
        return _chunk_data("Topic A", "Concept A"), FakeUsage()

    monkeypatch.setattr(svc, "_analyze_chunk", fake_analyze_chunk)
    monkeypatch.setattr(svc, "_chunk_text", _fixed_chunks("chunk a"))

    await svc.analyze_document("doc-1", "x" * 60)

    assert fake_supabase.count("topics", "insert") == 0  # nothing new to insert
    concept_rows = fake_supabase.rows("concepts")
    assert len(concept_rows) == 1
    assert concept_rows[0]["topic_id"] == "topic-existing"


@pytest.mark.asyncio
async def test_bulk_concept_insert_failure_falls_back_to_per_topic(monkeypatch):
    """One unwritable row must not cost the document every concept it
    extracted: the bulk insert is retried per topic so the blast radius stays
    a single topic, as it was before batching."""
    fake_supabase = FakeSupabase()
    fake_supabase.fail_next["concepts"] = True
    svc = _make_service(fake_supabase, monkeypatch)

    async def fake_analyze_chunk(document_id, text_chunk, chunk_index, total_chunks, semaphore):
        return {
            "topics": [
                {"name": "Topic A", "concepts": [{"name": "C1", "explanation": "e", "source_text": "s"}]},
                {"name": "Topic B", "concepts": [{"name": "C2", "explanation": "e", "source_text": "s"}]},
            ]
        }, FakeUsage()

    monkeypatch.setattr(svc, "_analyze_chunk", fake_analyze_chunk)
    monkeypatch.setattr(svc, "_chunk_text", _fixed_chunks("chunk a"))

    await svc.analyze_document("doc-1", "x" * 60)

    assert fake_supabase.count("concepts", "insert_failed") == 1  # the bulk attempt
    assert fake_supabase.count("concepts", "insert") == 2  # one per topic
    saved = fake_supabase.rows("concepts")
    assert {r["name"] for r in saved} == {"C1", "C2"}


@pytest.mark.asyncio
async def test_save_round_trips_do_not_grow_with_topic_count(monkeypatch):
    """The point of the batching: saving is a fixed number of round trips no
    matter how many topics and chunks the document produced. Per-topic saving
    cost 2 sequential trips each, which dominated processing time."""
    fake_supabase = FakeSupabase()
    svc = _make_service(fake_supabase, monkeypatch)

    async def fake_analyze_chunk(document_id, text_chunk, chunk_index, total_chunks, semaphore):
        return {
            "topics": [
                {
                    "name": f"Topic {chunk_index}-{t}",
                    "concepts": [
                        {"name": f"C{chunk_index}-{t}-{c}", "explanation": "e", "source_text": "s"}
                        for c in range(4)
                    ],
                }
                for t in range(3)
            ]
        }, FakeUsage()

    monkeypatch.setattr(svc, "_analyze_chunk", fake_analyze_chunk)
    monkeypatch.setattr(svc, "_chunk_text", _fixed_chunks(*[f"chunk {i}" for i in range(4)]))

    await svc.analyze_document("doc-1", "x" * 60)

    # 12 topics, 48 concepts, 4 chunks — still one trip each.
    assert fake_supabase.count("topics", "select") == 1
    assert fake_supabase.count("topics", "insert") == 1
    assert fake_supabase.count("concepts", "insert") == 1
    assert fake_supabase.count("llm_logs", "insert") == 1
    assert len(fake_supabase.calls) == 4

    assert len(fake_supabase.rows("topics")) == 12
    assert len(fake_supabase.rows("concepts")) == 48
    assert len(fake_supabase.rows("llm_logs")) == 4  # one row per chunk, one insert
