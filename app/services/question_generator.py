import logging
import asyncio
from typing import List, Dict, Any, Optional
from anthropic import AsyncAnthropic, APITimeoutError, APIConnectionError
from app.core.config import get_settings
from app.schemas.quiz import GeneratedQuestion, QuestionOption

settings = get_settings()

# Timeout configuration
CLAUDE_TIMEOUT_SECONDS = 90  # 90 seconds for Sonnet question generation (more complex)
MAX_API_RETRIES = 2

TOOL_NAME = "submit_question"


class QuestionGenerator:
    """
    Generates high-quality exam questions using Claude Sonnet.
    Focuses on Bloom's Taxonomy levels and real exam scenarios.

    Questions are generated via a forced tool call rather than free-text JSON:
    the schema carries a single `correct_index` instead of a per-option
    `is_correct` flag, so "0 correct" / "2 correct" responses are structurally
    impossible rather than merely discouraged by the prompt.
    """

    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = "claude-sonnet-4-6"  # Latest Sonnet for quality

    async def generate_questions_for_concept(
        self,
        concept_id: str,
        concept_name: str,
        concept_explanation: str,
        source_text: str,
        num_questions: int = 3,
        question_format: str = "standard",
        excluded_questions: List[str] | None = None,
    ) -> List[GeneratedQuestion]:
        """
        Generate multiple high-quality questions for a single concept.

        Args:
            concept_id: UUID of the concept
            concept_name: Name of the concept
            concept_explanation: Brief explanation of the concept
            source_text: Original text from document explaining this concept
            num_questions: Number of questions to generate (default 3)
            excluded_questions: Question stems already shown to the user — model
                will avoid repeating or closely paraphrasing these.

        Returns:
            List of GeneratedQuestion objects
        """
        try:
            # Determine difficulty distribution based on number of questions
            difficulties = self._get_difficulty_distribution(num_questions)

            logging.info(f"Generating {num_questions} questions for concept: {concept_name}")

            questions = []
            for i, difficulty in enumerate(difficulties):
                question = await self._generate_single_question(
                    concept_id=concept_id,
                    concept_name=concept_name,
                    concept_explanation=concept_explanation,
                    source_text=source_text,
                    difficulty=difficulty,
                    question_number=i + 1,
                    total_questions=num_questions,
                    question_format=question_format,
                    excluded_questions=excluded_questions or [],
                )

                if question:
                    questions.append(question)

            logging.info(f"Successfully generated {len(questions)}/{num_questions} questions for {concept_name}")
            return questions

        except Exception as e:
            logging.error(f"Failed to generate questions for concept {concept_name}: {e}")
            return []

    def _get_difficulty_distribution(self, num_questions: int) -> List[str]:
        """
        Determine difficulty distribution based on number of questions.
        Ensures balanced coverage across difficulty levels.
        """
        if num_questions == 1:
            return ["medium"]
        elif num_questions == 2:
            return ["easy", "hard"]
        elif num_questions == 3:
            return ["easy", "medium", "hard"]
        elif num_questions == 4:
            return ["easy", "medium", "medium", "hard"]
        elif num_questions == 5:
            return ["easy", "easy", "medium", "hard", "hard"]
        else:
            # For 6+ questions, distribute evenly
            easy_count = num_questions // 3
            hard_count = num_questions // 3
            medium_count = num_questions - easy_count - hard_count
            return (["easy"] * easy_count +
                   ["medium"] * medium_count +
                   ["hard"] * hard_count)

    async def _generate_single_question(
        self,
        concept_id: str,
        concept_name: str,
        concept_explanation: str,
        source_text: str,
        difficulty: str,
        question_number: int,
        total_questions: int,
        question_format: str = "standard",
        excluded_questions: List[str] | None = None,
    ) -> GeneratedQuestion | None:
        """
        Generate a single question via a forced tool call, with a bounded
        number of attempts. A validation failure sends the specific error back
        to the model as a tool_result so the next attempt is a targeted
        correction, not a blind resend of the same prompt.
        """
        system_prompt = self._build_system_prompt(difficulty, question_format)
        user_message = self._build_user_message(
            concept_name=concept_name,
            concept_explanation=concept_explanation,
            source_text=source_text,
            difficulty=difficulty,
            question_number=question_number,
            total_questions=total_questions,
            question_format=question_format,
            excluded_questions=excluded_questions or [],
        )
        tool_schema = self._build_tool_schema(question_format)
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_message}]

        for attempt in range(MAX_API_RETRIES + 1):
            try:
                response = await asyncio.wait_for(
                    self.client.messages.create(
                        model=self.model,
                        max_tokens=2000,
                        system=system_prompt,
                        messages=messages,
                        temperature=0.3,  # Balance creativity and consistency
                        tools=[tool_schema],
                        tool_choice={"type": "tool", "name": TOOL_NAME},
                    ),
                    timeout=CLAUDE_TIMEOUT_SECONDS
                )

                tool_use_block = next(
                    (b for b in response.content if b.type == "tool_use"), None
                )
                if tool_use_block is None:
                    raise ValueError("Model response did not include a tool call")

                question_data = tool_use_block.input

                error: Optional[str] = None
                question: GeneratedQuestion | None = None
                try:
                    question = self._create_question_object(
                        question_data, concept_id, difficulty, question_format
                    )
                except (KeyError, ValueError, TypeError) as e:
                    error = str(e)

                if question is not None:
                    error = self._validate_question_quality(question)

                if question is not None and error is None:
                    return question

                logging.warning(f"Question validation failed (attempt {attempt + 1}): {error}")
                if attempt < MAX_API_RETRIES:
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": (
                                f"Your submission was invalid: {error}. "
                                f"Call {TOOL_NAME} again with a corrected version "
                                "that fixes this specific issue."
                            ),
                            "is_error": True,
                        }],
                    })
                    continue

            except asyncio.TimeoutError:
                logging.error(f"Claude API timeout after {CLAUDE_TIMEOUT_SECONDS}s (attempt {attempt + 1})")
                if attempt < MAX_API_RETRIES:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue

            except (APITimeoutError, APIConnectionError) as e:
                logging.error(f"Claude API connection error (attempt {attempt + 1}): {e}")
                if attempt < MAX_API_RETRIES:
                    await asyncio.sleep(2 ** attempt)
                    continue

            except Exception as e:
                logging.error(f"Error generating question (attempt {attempt + 1}): {e}")
                if attempt < MAX_API_RETRIES:
                    continue

        return None

    def _build_tool_schema(self, question_format: str = "standard") -> Dict[str, Any]:
        """
        Tool (function-call) schema for question submission.

        Correctness is carried as a single `correct_index` into `options`
        rather than a per-option `is_correct` flag, so "0 correct" / "2+
        correct" responses — the exact failure mode seen in production — are
        structurally impossible to express, not just discouraged by the prompt.
        """
        num_options = 3 if question_format == "explanatory" else 4
        option_text_desc = (
            "A 50-150 word paragraph explaining a position or reasoning."
            if question_format == "explanatory"
            else "A concise, single-sentence answer option."
        )

        properties: Dict[str, Any] = {
            "question": {
                "type": "string",
                "description": "Clear, specific question text (20-500 characters).",
            },
            "options": {
                "type": "array",
                "minItems": num_options,
                "maxItems": num_options,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": option_text_desc},
                        "explanation": {
                            "type": "string",
                            "description": "Why this option is correct or incorrect in this specific case (>= 10 characters).",
                        },
                    },
                    "required": ["text", "explanation"],
                },
                "description": f"Exactly {num_options} answer options, in the order they should be presented.",
            },
            "correct_index": {
                "type": "integer",
                "minimum": 0,
                "maximum": num_options - 1,
                "description": f"0-based index into `options` of the single correct answer (0 to {num_options - 1}).",
            },
            "hint": {
                "type": "string",
                "description": "A subtle hint that doesn't give away the answer.",
            },
        }
        required = ["question", "options", "correct_index", "hint"]

        if question_format == "explanatory":
            properties["post_answer_summary"] = {
                "type": "string",
                "description": "2-3 sentence canonical textbook explanation, shown after the student answers.",
            }
            required.append("post_answer_summary")

        return {
            "name": TOOL_NAME,
            "description": "Submit one multiple-choice exam question with exactly one correct answer.",
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _build_system_prompt(self, difficulty: str, question_format: str = "standard") -> str:
        """Build the system prompt for question generation."""
        difficulty_guides = {
            "easy": (
                "EASY questions test basic recall and understanding.\n"
                "- Focus on definitions, key terms, and fundamental concepts\n"
                "- Straightforward language\n"
                "- Direct questions with clear answers"
            ),
            "medium": (
                "MEDIUM questions test comprehension and application.\n"
                "- Require understanding concepts, not just memorization\n"
                "- Apply knowledge to similar scenarios\n"
                "- Connect related ideas"
            ),
            "hard": (
                "HARD questions test analysis, evaluation, and synthesis.\n"
                "- Require deep understanding and critical thinking\n"
                "- Apply concepts to novel situations\n"
                "- Analyze relationships and make judgments"
            )
        }

        format_instructions = ""
        if question_format == "explanatory":
            format_instructions = """
EXPLANATORY FORMAT RULES:
- Generate EXACTLY 3 options (not 4).
- Each option is a 50-150 word paragraph explaining a position or reasoning.
- The other two options should encode common misconceptions.
- Options should be teaching the concept while the student evaluates them.
"""
        else:
            format_instructions = """
STANDARD FORMAT RULES:
- EXACTLY 4 options (A, B, C, D).
- Options are concise (single sentences).
"""

        return f"""You are an expert educational assessment designer specializing in creating high-quality exam questions.

Your task is to create ONE multiple-choice question based on the provided concept and source material, then submit it via the {TOOL_NAME} tool.

DIFFICULTY LEVEL: {difficulty.upper()}
{difficulty_guides[difficulty]}

{format_instructions}

CRITICAL QUALITY REQUIREMENTS:
1. **Clear and Unambiguous**: Question should have ONE definitively correct answer
2. **Exam-Ready**: Written at the level of professional certification or university exams
3. **Grounded in Source**: Use the actual content from the provided material
4. **Realistic Distractors**: Wrong answers should be plausible but clearly incorrect
5. **Educational Value**: Test meaningful understanding, not trivial details

RULES:
- NO "All of the above" or "None of the above"
- NO "Both A and B" compound options"""

    def _build_user_message(
        self,
        concept_name: str,
        concept_explanation: str,
        source_text: str,
        difficulty: str,
        question_number: int,
        total_questions: int,
        question_format: str = "standard",
        excluded_questions: List[str] | None = None,
    ) -> str:
        """Build the user message with concept details."""
        exclusion_block = ""
        if excluded_questions:
            stems = "\n".join(
                f"- {q[:120]}" for q in excluded_questions[:20]
            )
            exclusion_block = (
                f"\nAVOID questions similar to any of these already-shown questions "
                f"(do not repeat or closely paraphrase them):\n{stems}\n"
            )

        return f"""Create question {question_number} of {total_questions} ({difficulty} difficulty):{exclusion_block}

CONCEPT: {concept_name}

CONCEPT EXPLANATION:
{concept_explanation}

SOURCE MATERIAL:
{source_text[:1500]}

Generate ONE high-quality {difficulty}-level multiple-choice question in {question_format} format that tests this concept."""

    def _create_question_object(
        self,
        question_data: Dict[str, Any],
        concept_id: str,
        difficulty: str,
        question_format: str = "standard"
    ) -> GeneratedQuestion:
        """Convert the tool call's parsed input into a GeneratedQuestion object."""
        options_data = question_data.get("options", [])

        expected_options = 3 if question_format == "explanatory" else 4
        if len(options_data) != expected_options:
            raise ValueError(f"Expected {expected_options} options, got {len(options_data)}")

        correct_index = question_data.get("correct_index")
        if not isinstance(correct_index, int) or not (0 <= correct_index < len(options_data)):
            raise ValueError(
                f"correct_index {correct_index!r} is out of range for {len(options_data)} options"
            )

        options = [
            QuestionOption(
                option_text=opt.get("text", ""),
                option_index=i,
                is_correct=(i == correct_index),
                explanation=opt.get("explanation", ""),
            )
            for i, opt in enumerate(options_data)
        ]

        return GeneratedQuestion(
            question=question_data.get("question", ""),
            options=options,
            hint=question_data.get("hint"),
            difficulty_level=difficulty,  # type: ignore[arg-type]
            concept_id=concept_id,
            question_format=question_format,  # type: ignore[arg-type]
            post_answer_summary=question_data.get("post_answer_summary")
        )

    def _validate_question_quality(self, question: GeneratedQuestion) -> Optional[str]:
        """
        Validate question meets quality standards.
        Returns None if valid, or a description of the problem otherwise —
        the description is fed back to the model as a targeted correction.
        """
        if len(question.question) < 20 or len(question.question) > 500:
            return f"Question text length invalid: {len(question.question)} characters (must be 20-500)"

        for i, opt in enumerate(question.options):
            if not opt.option_text or len(opt.option_text) < 3:
                return f"Option {i} text is too short or empty"
            if not opt.explanation or len(opt.explanation) < 10:
                return f"Option {i} explanation is too short or empty (must be at least 10 characters)"

        option_texts = [opt.option_text.lower() for opt in question.options]
        if len(set(option_texts)) != len(option_texts):
            return "Two or more options have duplicate text"

        if question.hint and len(question.hint) < 10:
            return "Hint is too short"

        return None

    def calculate_questions_per_concept(
        self,
        concept_explanation: str,
        source_text: str,
        min_questions: int = 2,
        max_questions: int = 5
    ) -> int:
        """
        Dynamically determine number of questions based on content richness.

        Args:
            concept_explanation: The concept's explanation
            source_text: Original source material
            min_questions: Minimum questions to generate
            max_questions: Maximum questions to generate

        Returns:
            Number of questions to generate (between min and max)
        """
        # Calculate content richness score
        content_length = len(source_text) + len(concept_explanation)

        # More content = more questions (simple heuristic)
        if content_length < 200:
            return min_questions
        elif content_length < 500:
            return min(3, max_questions)
        elif content_length < 1000:
            return min(4, max_questions)
        else:
            return max_questions
