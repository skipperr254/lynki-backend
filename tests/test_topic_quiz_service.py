"""
Tests for topic_quiz_service's per-concept parallel generation and wall-clock
budget. Mirrors test_on_demand_quiz_service.py's approach: monkeypatch the
QuestionGenerator call so these exercise only the orchestration logic, no
live API or DB calls.
"""
import asyncio

import pytest

import app.services.topic_quiz_service as svc
from app.schemas.quiz import GeneratedQuestion, QuestionOption


def _fake_question(concept_id: str) -> GeneratedQuestion:
    return GeneratedQuestion(
        question=f"What does concept {concept_id} do in this system?",
        options=[
            QuestionOption(option_text=f"Option {i}", option_index=i, is_correct=(i == 0),
                            explanation=f"Explanation {i} in enough detail to pass validation.")
            for i in range(4)
        ],
        hint="Think about it.",
        difficulty_level="medium",
        concept_id=concept_id,
    )


class FakeDB:
    def __init__(self, topic_name, concepts):
        self._topic_name = topic_name
        self._concepts = concepts

    def table(self, name):
        return FakeQuery(name, self)


class FakeQuery:
    def __init__(self, table_name, db):
        self._table_name = table_name
        self._db = db
        self._payload = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def single(self):
        return self

    def insert(self, payload):
        self._payload = payload
        return self

    def execute(self):
        class _Resp:
            pass
        resp = _Resp()
        if self._table_name == "topics":
            resp.data = {"name": self._db._topic_name}
        elif self._table_name == "concepts":
            resp.data = self._db._concepts
        elif self._table_name == "topic_quiz_sessions":
            # Echo the inserted row back, like Supabase's insert().execute()
            # does, plus the DB-generated id the real schema's default supplies.
            resp.data = [{"id": "session-1", **self._payload}]
        return resp


def _concepts(n):
    return [{"id": f"concept-{i}", "name": f"Concept {i}", "explanation": "e", "source_text": "s"} for i in range(n)]


def _fake_generator_class(generate_fn):
    """A stand-in for QuestionGenerator that skips the real AsyncAnthropic
    client construction entirely (construction alone costs ~1s on this
    machine — noise unrelated to the concurrency/timeout behavior under test)."""
    class _Fake:
        def __init__(self):
            pass

        generate_questions_for_concept = generate_fn

    return _Fake


@pytest.mark.asyncio
async def test_generation_runs_concepts_in_parallel_not_sequentially(monkeypatch):
    concepts = _concepts(3)
    db = FakeDB("Test Topic", concepts)

    async def fake_generate(self, concept_id, concept_name, concept_explanation,
                             source_text, num_questions=3, question_format="standard",
                             excluded_questions=None):
        await asyncio.sleep(0.2)  # each concept "takes" 200ms
        return [_fake_question(concept_id)]

    monkeypatch.setattr(svc, "QuestionGenerator", _fake_generator_class(fake_generate))

    start = asyncio.get_event_loop().time()
    result = await svc.TopicQuizService._generate_session(
        db, user_id="u1", course_id="c1", topic_id="topic-1", question_format="standard"
    )
    elapsed = asyncio.get_event_loop().time() - start

    # Sequential would take >= 3 * 0.2s = 0.6s; parallel should be close to 0.2s.
    assert elapsed < 0.5
    assert result["total_questions"] == 3


@pytest.mark.asyncio
async def test_generation_respects_timeout_and_keeps_partial_results(monkeypatch):
    monkeypatch.setattr(svc, "TOPIC_QUIZ_GENERATION_TIMEOUT", 0.1)
    concepts = _concepts(2)
    db = FakeDB("Test Topic", concepts)

    async def fake_generate(self, concept_id, concept_name, concept_explanation,
                             source_text, num_questions=3, question_format="standard",
                             excluded_questions=None):
        if concept_id == "concept-1":
            await asyncio.sleep(10)  # never finishes within the budget
        return [_fake_question(concept_id)]

    monkeypatch.setattr(svc, "QuestionGenerator", _fake_generator_class(fake_generate))

    start = asyncio.get_event_loop().time()
    result = await svc.TopicQuizService._generate_session(
        db, user_id="u1", course_id="c1", topic_id="topic-1", question_format="standard"
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 5  # bounded by TOPIC_QUIZ_GENERATION_TIMEOUT, not the 10s sleep
    assert result["total_questions"] == 1  # only concept-0 made it in time


@pytest.mark.asyncio
async def test_no_questions_at_all_raises_value_error(monkeypatch):
    concepts = _concepts(1)
    db = FakeDB("Test Topic", concepts)

    async def fake_generate(self, *args, **kwargs):
        return []

    monkeypatch.setattr(svc, "QuestionGenerator", _fake_generator_class(fake_generate))

    with pytest.raises(ValueError, match="no questions"):
        await svc.TopicQuizService._generate_session(
            db, user_id="u1", course_id="c1", topic_id="topic-1", question_format="standard"
        )
