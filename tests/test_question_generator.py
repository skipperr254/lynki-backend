"""
Tests for QuestionGenerator's tool-use based question generation.

No live Anthropic API calls are made — `client.messages.create` is replaced
with an AsyncMock returning fake tool_use responses. These tests exist to
guard the exact bug that used to reach production: the model returning 0 or
2+ "correct" options, and the old blind-retry loop resending an identical
prompt instead of correcting the specific problem.
"""
from unittest.mock import AsyncMock

import pytest

from app.schemas.quiz import GeneratedQuestion, QuestionOption
from app.services.question_generator import QuestionGenerator


class FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, input_data, block_id="tool_1"):
        self.input = input_data
        self.id = block_id
        self.name = "submit_question"


class FakeResponse:
    def __init__(self, content):
        self.content = content


def valid_input(correct_index: int = 1) -> dict:
    return {
        "question": "What is the primary purpose of a load balancer in a distributed system?",
        "options": [
            {
                "text": "To encrypt traffic between services",
                "explanation": "Encryption is TLS's job, not a load balancer's core purpose.",
            },
            {
                "text": "To distribute incoming requests across multiple servers",
                "explanation": "This is the defining function of a load balancer.",
            },
            {
                "text": "To store application state persistently",
                "explanation": "Persistent storage is handled by databases, not load balancers.",
            },
            {
                "text": "To compile source code at deploy time",
                "explanation": "Compilation happens in the build pipeline, unrelated to load balancing.",
            },
        ],
        "correct_index": correct_index,
        "hint": "Think about what happens when traffic spikes on a single server.",
    }


def make_generator() -> QuestionGenerator:
    gen = QuestionGenerator()
    gen.client.messages.create = AsyncMock()
    return gen


@pytest.mark.asyncio
async def test_valid_tool_response_produces_exactly_one_correct_option():
    gen = make_generator()
    gen.client.messages.create.return_value = FakeResponse(
        [FakeToolUseBlock(valid_input(correct_index=2))]
    )

    question = await gen._generate_single_question(
        concept_id="c1",
        concept_name="Load balancing",
        concept_explanation="Distributes traffic across servers.",
        source_text="A load balancer sits in front of a pool of servers...",
        difficulty="medium",
        question_number=1,
        total_questions=1,
    )

    assert question is not None
    correct = [o for o in question.options if o.is_correct]
    assert len(correct) == 1
    assert correct[0].option_index == 2
    assert gen.client.messages.create.await_count == 1


@pytest.mark.asyncio
async def test_quality_failure_triggers_targeted_correction_not_blind_retry():
    gen = make_generator()

    bad = valid_input(correct_index=1)
    bad["options"][1]["explanation"] = "ok"  # too short — fails quality check

    gen.client.messages.create.side_effect = [
        FakeResponse([FakeToolUseBlock(bad, block_id="tool_1")]),
        FakeResponse([FakeToolUseBlock(valid_input(correct_index=1), block_id="tool_2")]),
    ]

    question = await gen._generate_single_question(
        concept_id="c1",
        concept_name="Load balancing",
        concept_explanation="Distributes traffic across servers.",
        source_text="A load balancer sits in front of a pool of servers...",
        difficulty="medium",
        question_number=1,
        total_questions=1,
    )

    assert question is not None
    assert gen.client.messages.create.await_count == 2

    # The second call must carry the specific validation error back to the
    # model as a tool_result, not just resend the original user message.
    second_call_messages = gen.client.messages.create.await_args_list[1].kwargs["messages"]
    assert len(second_call_messages) == 3
    tool_result = second_call_messages[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "tool_1"
    assert "too short" in tool_result["content"]


@pytest.mark.asyncio
async def test_persistent_invalid_correct_index_exhausts_retries_and_returns_none():
    gen = make_generator()
    # correct_index out of range on every attempt — structurally invalid,
    # should never happen with the schema in place, but must fail safely.
    gen.client.messages.create.return_value = FakeResponse(
        [FakeToolUseBlock(valid_input(correct_index=99))]
    )

    question = await gen._generate_single_question(
        concept_id="c1",
        concept_name="Load balancing",
        concept_explanation="Distributes traffic across servers.",
        source_text="A load balancer sits in front of a pool of servers...",
        difficulty="medium",
        question_number=1,
        total_questions=1,
    )

    assert question is None
    from app.services.question_generator import MAX_API_RETRIES
    assert gen.client.messages.create.await_count == MAX_API_RETRIES + 1


def test_tool_schema_reflects_option_count_and_correctness_by_construction():
    gen = QuestionGenerator()

    standard = gen._build_tool_schema("standard")
    props = standard["input_schema"]["properties"]
    assert props["options"]["minItems"] == 4
    assert props["options"]["maxItems"] == 4
    assert props["correct_index"]["maximum"] == 3
    assert "is_correct" not in props["options"]["items"]["properties"]
    assert "post_answer_summary" not in standard["input_schema"]["required"]

    explanatory = gen._build_tool_schema("explanatory")
    props_e = explanatory["input_schema"]["properties"]
    assert props_e["options"]["minItems"] == 3
    assert props_e["options"]["maxItems"] == 3
    assert props_e["correct_index"]["maximum"] == 2
    assert "post_answer_summary" in explanatory["input_schema"]["required"]


@pytest.mark.parametrize(
    "mutate,expected_substring",
    [
        (lambda q: setattr(q, "question", "Too short?"), "length invalid"),
        (lambda q: setattr(q.options[0], "option_text", ""), "too short or empty"),
        (lambda q: setattr(q.options[0], "explanation", "short"), "explanation is too short"),
        (
            lambda q: [setattr(o, "option_text", "Same text") for o in q.options],
            "duplicate",
        ),
        (lambda q: setattr(q, "hint", "short"), "Hint is too short"),
    ],
)
def test_validate_question_quality_reports_specific_reason(mutate, expected_substring):
    gen = QuestionGenerator()
    question = GeneratedQuestion(
        question="What does a load balancer primarily do in a distributed system?",
        options=[
            QuestionOption(option_text=f"Option {i}", option_index=i, is_correct=(i == 0),
                            explanation=f"Explanation for option {i} in enough detail.")
            for i in range(4)
        ],
        hint="Think about traffic distribution across servers.",
        difficulty_level="medium",
        concept_id="c1",
    )
    mutate(question)

    error = gen._validate_question_quality(question)
    assert error is not None
    assert expected_substring.lower() in error.lower()


def test_validate_question_quality_passes_for_well_formed_question():
    gen = QuestionGenerator()
    question = GeneratedQuestion(
        question="What does a load balancer primarily do in a distributed system?",
        options=[
            QuestionOption(option_text=f"Option {i}", option_index=i, is_correct=(i == 0),
                            explanation=f"Explanation for option {i} in enough detail.")
            for i in range(4)
        ],
        hint="Think about traffic distribution across servers.",
        difficulty_level="medium",
        concept_id="c1",
    )
    assert gen._validate_question_quality(question) is None
