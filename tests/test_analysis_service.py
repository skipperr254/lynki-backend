"""
Tests for analysis_service's single-call whole-document analysis and batched
save: that the complete text goes out in exactly one API call, that terminal
failures (oversized input, truncated output) raise without writing a partial
structure, that transport failures retry, and — unchanged from when this
service chunked — the topic dedup, reuse of a reprocessed document's existing
topics, the per-topic fallback when a bulk concept insert is rejected, and the
round-trip count the batching exists to hold down.

`client.messages.parse` is faked so these tests exercise only the
orchestration logic in `analyze_document` — no live API calls.
`AnalysisService.__init__` is bypassed (it constructs a real AsyncAnthropic
client) via `__new__`, matching this repo's existing pattern of avoiding that
~1s construction cost in tests that don't need it (see
test_topic_quiz_service.py's `_fake_generator_class` comment).
"""
import asyncio

import pytest

from app.services.analysis_service import (
    AnalysisService,
    ConceptOut,
    DocumentStructure,
    MAX_ANALYSIS_CHARS,
    MAX_API_RETRIES,
    TopicOut,
)


class FakeUsage:
    input_tokens = 10
    output_tokens = 10


class FakeParsedResponse:
    """Stands in for the object `messages.parse` returns."""

    def __init__(self, structure, stop_reason: str = "end_turn"):
        self.parsed_output = structure
        self.stop_reason = stop_reason
        self.usage = FakeUsage()


class FakeMessages:
    """Records every parse() call so tests can assert how many were made and
    what text each one actually received. Queued entries that are exceptions
    are raised instead of returned, which is how the retry paths are driven."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("parse() called more times than responses queued")
        nxt = self._responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


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


def _make_service(fake_supabase, monkeypatch, responses=()) -> AnalysisService:
    svc = AnalysisService.__new__(AnalysisService)
    svc.supabase = fake_supabase
    svc.client = FakeClient(responses)
    svc.model = "claude-sonnet-5"

    async def _run(fn):
        # A real yield point (unlike a bare `return fn()`) mirrors the real
        # run_db_operation, which awaits a thread-pool executor.
        await asyncio.sleep(0)
        return fn()

    monkeypatch.setattr("app.services.analysis_service.run_db_operation", _run)
    return svc


@pytest.fixture
def no_backoff(monkeypatch):
    """Collapse the retry backoff (1s, then 2s) so retry tests stay fast.
    `real_sleep` is captured before patching, so there's no recursion."""
    real_sleep = asyncio.sleep

    async def _fast(_delay):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _fast)


def _structure(*topics) -> DocumentStructure:
    """Build a DocumentStructure from (topic_name, [concept_name, ...]) pairs."""
    return DocumentStructure(
        topics=[
            TopicOut(
                name=name,
                concepts=[
                    ConceptOut(name=c, explanation="e", source_text="s")
                    for c in concept_names
                ],
            )
            for name, concept_names in topics
        ]
    )


@pytest.mark.asyncio
async def test_whole_text_goes_out_in_exactly_one_call(monkeypatch):
    """The point of the change: no chunking. One call, carrying the complete
    text rather than a ~8000-char slice of it."""
    fake_supabase = FakeSupabase()
    text = "word " * 10_000  # 50k chars — would have been ~7 chunks before
    svc = _make_service(
        fake_supabase,
        monkeypatch,
        [FakeParsedResponse(_structure(("Cell Biology", ["Mitosis"])))],
    )

    await svc.analyze_document("doc-1", text)

    calls = svc.client.messages.calls
    assert len(calls) == 1, "document must be analysed in a single call"
    assert calls[0]["messages"][0]["content"] == text, "full text must be sent intact"
    assert calls[0]["output_format"] is DocumentStructure


@pytest.mark.asyncio
async def test_topics_and_concepts_are_saved(monkeypatch):
    fake_supabase = FakeSupabase()
    svc = _make_service(
        fake_supabase,
        monkeypatch,
        [
            FakeParsedResponse(
                _structure(
                    ("Cell Biology", ["Mitosis", "Meiosis"]),
                    ("Genetics", ["Alleles"]),
                )
            )
        ],
    )

    await svc.analyze_document("doc-1", "x" * 60)

    topics = fake_supabase.rows("topics")
    assert {t["name"] for t in topics} == {"Cell Biology", "Genetics"}

    concepts = fake_supabase.rows("concepts")
    assert {c["name"] for c in concepts} == {"Mitosis", "Meiosis", "Alleles"}
    assert all(c["complexity_level"] == "intermediate" for c in concepts)


@pytest.mark.asyncio
async def test_repeated_topic_name_collapses_to_one_row(monkeypatch):
    """The model naming the same topic twice must still yield one topic row,
    carrying both entries' concepts."""
    fake_supabase = FakeSupabase()
    svc = _make_service(
        fake_supabase,
        monkeypatch,
        [
            FakeParsedResponse(
                _structure(
                    ("Cell Biology", ["Mitosis"]),
                    ("Cell Biology", ["Meiosis"]),
                )
            )
        ],
    )

    await svc.analyze_document("doc-1", "x" * 60)

    topics = fake_supabase.rows("topics")
    assert len(topics) == 1 and topics[0]["name"] == "Cell Biology"

    concepts = fake_supabase.rows("concepts")
    assert {c["name"] for c in concepts} == {"Mitosis", "Meiosis"}
    assert len({c["topic_id"] for c in concepts}) == 1


@pytest.mark.asyncio
async def test_text_too_short_returns_without_calling_the_api(monkeypatch):
    fake_supabase = FakeSupabase()
    svc = _make_service(fake_supabase, monkeypatch, [])

    await svc.analyze_document("doc-1", "too short")

    assert svc.client.messages.calls == []
    assert fake_supabase.calls == []


@pytest.mark.asyncio
async def test_oversized_text_raises_before_any_api_call(monkeypatch):
    """Oversized documents fail with user-facing prose rather than being
    truncated or silently re-split."""
    fake_supabase = FakeSupabase()
    svc = _make_service(fake_supabase, monkeypatch, [])

    with pytest.raises(ValueError, match="too large"):
        await svc.analyze_document("doc-1", "x" * (MAX_ANALYSIS_CHARS + 1))

    assert svc.client.messages.calls == [], "must not spend a call on oversized input"
    assert fake_supabase.calls == []


@pytest.mark.asyncio
async def test_truncated_response_raises_and_saves_nothing(monkeypatch):
    """A max_tokens stop used to be logged and the partial output parsed
    anyway. With one call per document that would silently drop the back half
    of the material, so it must raise instead."""
    fake_supabase = FakeSupabase()
    svc = _make_service(
        fake_supabase,
        monkeypatch,
        [
            FakeParsedResponse(
                _structure(("Cell Biology", ["Mitosis"])), stop_reason="max_tokens"
            )
        ],
    )

    with pytest.raises(ValueError, match="one pass"):
        await svc.analyze_document("doc-1", "x" * 60)

    assert fake_supabase.calls == [], "no partial structure may be written"


@pytest.mark.asyncio
async def test_timeout_then_success_saves_normally(monkeypatch, no_backoff):
    fake_supabase = FakeSupabase()
    svc = _make_service(
        fake_supabase,
        monkeypatch,
        [
            asyncio.TimeoutError(),
            FakeParsedResponse(_structure(("Cell Biology", ["Mitosis"]))),
        ],
    )

    await svc.analyze_document("doc-1", "x" * 60)

    assert len(svc.client.messages.calls) == 2
    assert [c["name"] for c in fake_supabase.rows("concepts")] == ["Mitosis"]


@pytest.mark.asyncio
async def test_retry_exhaustion_raises_and_touches_nothing(monkeypatch, no_backoff):
    fake_supabase = FakeSupabase()
    svc = _make_service(
        fake_supabase,
        monkeypatch,
        [asyncio.TimeoutError() for _ in range(MAX_API_RETRIES + 1)],
    )

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        await svc.analyze_document("doc-1", "x" * 60)

    assert len(svc.client.messages.calls) == MAX_API_RETRIES + 1
    assert fake_supabase.calls == []


@pytest.mark.asyncio
async def test_reprocessed_document_reuses_existing_topic_rows(monkeypatch):
    """A document reprocessed after a partial failure already has topic rows;
    reusing them is what stops a retry duplicating every topic."""
    fake_supabase = FakeSupabase()
    fake_supabase.seed_topic("doc-1", "Cell Biology", "existing-topic-id")
    svc = _make_service(
        fake_supabase,
        monkeypatch,
        [FakeParsedResponse(_structure(("Cell Biology", ["Mitosis"])))],
    )

    await svc.analyze_document("doc-1", "x" * 60)

    assert fake_supabase.count("topics", "insert") == 0, "must reuse, not re-insert"
    concepts = fake_supabase.rows("concepts")
    assert [c["topic_id"] for c in concepts] == ["existing-topic-id"]


@pytest.mark.asyncio
async def test_bulk_concept_insert_failure_falls_back_to_per_topic(monkeypatch):
    """One unwritable row must cost its own topic's concepts, not the
    document's."""
    fake_supabase = FakeSupabase()
    fake_supabase.fail_next["concepts"] = True
    svc = _make_service(
        fake_supabase,
        monkeypatch,
        [
            FakeParsedResponse(
                _structure(
                    ("Cell Biology", ["Mitosis"]),
                    ("Genetics", ["Alleles"]),
                )
            )
        ],
    )

    await svc.analyze_document("doc-1", "x" * 60)

    assert fake_supabase.count("concepts", "insert_failed") == 1
    # One INSERT per topic after the bulk attempt was rejected.
    assert fake_supabase.count("concepts", "insert") == 2
    assert {c["name"] for c in fake_supabase.rows("concepts")} == {"Mitosis", "Alleles"}


@pytest.mark.asyncio
async def test_save_round_trips_do_not_grow_with_topic_count(monkeypatch):
    """The batching exists to hold the save at a fixed 3 round trips (one
    topic SELECT, one topic INSERT, one concept INSERT) no matter how many
    topics the document produced."""
    fake_supabase = FakeSupabase()
    svc = _make_service(
        fake_supabase,
        monkeypatch,
        [FakeParsedResponse(_structure(*[(f"Topic {i}", [f"C{i}"]) for i in range(12)]))],
    )

    await svc.analyze_document("doc-1", "x" * 60)

    assert fake_supabase.count("topics", "select") == 1
    assert fake_supabase.count("topics", "insert") == 1
    assert fake_supabase.count("concepts", "insert") == 1
    assert len(fake_supabase.rows("topics")) == 12


@pytest.mark.asyncio
async def test_one_llm_logs_row_per_document(monkeypatch):
    """Was one row per chunk; a single call means a single row."""
    fake_supabase = FakeSupabase()
    svc = _make_service(
        fake_supabase,
        monkeypatch,
        [FakeParsedResponse(_structure(("Cell Biology", ["Mitosis"])))],
    )

    await svc.analyze_document("doc-1", "x" * 60)

    assert fake_supabase.count("llm_logs", "insert") == 1
    rows = fake_supabase.rows("llm_logs")
    assert len(rows) == 1
    assert rows[0]["operation"] == "structure_extraction"
    assert rows[0]["model"] == "claude-sonnet-5"
