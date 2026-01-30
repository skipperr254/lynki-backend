import json
import logging
import re
from typing import List, Dict, Any, Tuple
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from app.core.config import get_settings
from app.schemas.quiz import GeneratedQuestion, QuestionOption

settings = get_settings()

class QuestionGenerator:
    """
    Generates high-quality exam questions using Claude Sonnet.
    Focuses on Bloom's Taxonomy levels and real exam scenarios.
    """
    
    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = "claude-sonnet-4-20250514"  # Latest Sonnet for quality
        
    async def generate_questions_for_concept(
        self,
        concept_id: str,
        concept_name: str,
        concept_explanation: str,
        source_text: str,
        num_questions: int = 3
    ) -> List[GeneratedQuestion]:
        """
        Generate multiple high-quality questions for a single concept.
        
        Args:
            concept_id: UUID of the concept
            concept_name: Name of the concept
            concept_explanation: Brief explanation of the concept
            source_text: Original text from document explaining this concept
            num_questions: Number of questions to generate (default 3)
            
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
                    total_questions=num_questions
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
        total_questions: int
    ) -> GeneratedQuestion | None:
        """Generate a single question with retries."""
        max_retries = 2
        
        for attempt in range(max_retries + 1):
            try:
                system_prompt = self._build_system_prompt(difficulty)
                user_message = self._build_user_message(
                    concept_name=concept_name,
                    concept_explanation=concept_explanation,
                    source_text=source_text,
                    difficulty=difficulty,
                    question_number=question_number,
                    total_questions=total_questions
                )
                
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                    temperature=0.3  # Balance creativity and consistency
                )
                
                # Extract text content
                content_block = response.content[0]
                if not isinstance(content_block, TextBlock):
                    raise ValueError(f"Unexpected content type: {type(content_block).__name__}")
                
                response_text = content_block.text
                
                # Parse and validate
                question_data = self._parse_question_response(response_text)
                question = self._create_question_object(question_data, concept_id, difficulty)
                
                # Validate quality
                if self._validate_question_quality(question):
                    return question
                else:
                    logging.warning(f"Question quality validation failed (attempt {attempt + 1})")
                    if attempt < max_retries:
                        continue
                    
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    continue
            except Exception as e:
                logging.error(f"Error generating question (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    continue
                    
        return None
    
    def _build_system_prompt(self, difficulty: str) -> str:
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
        
        return f"""You are an expert educational assessment designer specializing in creating high-quality exam questions.

Your task is to create ONE multiple-choice question based on the provided concept and source material.

DIFFICULTY LEVEL: {difficulty.upper()}
{difficulty_guides[difficulty]}

CRITICAL QUALITY REQUIREMENTS:
1. **Clear and Unambiguous**: Question should have ONE definitively correct answer
2. **Exam-Ready**: Written at the level of professional certification or university exams
3. **Grounded in Source**: Use the actual content from the provided material
4. **Realistic Distractors**: Wrong answers should be plausible but clearly incorrect
5. **Educational Value**: Test meaningful understanding, not trivial details

OUTPUT FORMAT (MUST be valid JSON):
{{
  "question": "Clear, specific question text (no vague wording)",
  "options": [
    {{
      "text": "Option A text",
      "is_correct": true,
      "explanation": "Detailed explanation of why this is correct, referencing source material"
    }},
    {{
      "text": "Option B text",
      "is_correct": false,
      "explanation": "Specific explanation of why this is incorrect and what misconception it represents"
    }},
    {{
      "text": "Option C text",
      "is_correct": false,
      "explanation": "Specific explanation of why this is incorrect"
    }},
    {{
      "text": "Option D text",
      "is_correct": false,
      "explanation": "Specific explanation of why this is incorrect"
    }}
  ],
  "hint": "Subtle hint that guides thinking without revealing the answer"
}}

RULES:
- EXACTLY 4 options (A, B, C, D)
- EXACTLY 1 correct answer
- NO "All of the above" or "None of the above"
- NO "Both A and B" compound options
- Each explanation must be 1-3 sentences
- Hint should prompt reflection, not give away the answer
- Output ONLY valid JSON, no markdown formatting"""
    
    def _build_user_message(
        self,
        concept_name: str,
        concept_explanation: str,
        source_text: str,
        difficulty: str,
        question_number: int,
        total_questions: int
    ) -> str:
        """Build the user message with concept details."""
        return f"""Create question {question_number} of {total_questions} ({difficulty} difficulty):

CONCEPT: {concept_name}

CONCEPT EXPLANATION:
{concept_explanation}

SOURCE MATERIAL:
{source_text[:1500]}

Generate ONE high-quality {difficulty}-level multiple-choice question that tests this concept."""
    
    def _parse_question_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and clean the JSON response."""
        # Remove markdown code blocks if present
        response_text = re.sub(r'^```(?:json)?\\s*', '', response_text, flags=re.MULTILINE)
        response_text = re.sub(r'\\s*```$', '', response_text, flags=re.MULTILINE)
        
        # Find JSON boundaries
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")
        
        json_str = response_text[start_idx:end_idx]
        
        # Clean common JSON issues
        json_str = re.sub(r',\\s*}', '}', json_str)
        json_str = re.sub(r',\\s*]', ']', json_str)
        
        return json.loads(json_str)
    
    def _create_question_object(
        self,
        question_data: Dict[str, Any],
        concept_id: str,
        difficulty: str
    ) -> GeneratedQuestion:
        """Convert parsed JSON to GeneratedQuestion object."""
        options_data = question_data.get("options", [])
        
        if len(options_data) != 4:
            raise ValueError(f"Expected 4 options, got {len(options_data)}")
        
        options = []
        for i, opt in enumerate(options_data):
            options.append(QuestionOption(
                option_text=opt["text"],
                option_index=i,
                is_correct=opt.get("is_correct", False),
                explanation=opt.get("explanation", "")
            ))
        
        # Validate exactly one correct answer
        correct_count = sum(1 for opt in options if opt.is_correct)
        if correct_count != 1:
            raise ValueError(f"Expected exactly 1 correct answer, got {correct_count}")
        
        return GeneratedQuestion(
            question=question_data["question"],
            options=options,
            hint=question_data.get("hint"),
            difficulty_level=difficulty,  # type: ignore[arg-type]
            concept_id=concept_id
        )
    
    def _validate_question_quality(self, question: GeneratedQuestion) -> bool:
        """
        Validate question meets quality standards.
        Returns True if valid, False otherwise.
        """
        # Check question text length (not too short or too long)
        if len(question.question) < 20 or len(question.question) > 500:
            logging.warning(f"Question text length invalid: {len(question.question)}")
            return False
        
        # Check all options have text
        for opt in question.options:
            if not opt.option_text or len(opt.option_text) < 3:
                logging.warning("Option text too short or empty")
                return False
            if not opt.explanation or len(opt.explanation) < 10:
                logging.warning("Option explanation too short or empty")
                return False
        
        # Check options are not too similar (basic check)
        option_texts = [opt.option_text.lower() for opt in question.options]
        if len(set(option_texts)) != len(option_texts):
            logging.warning("Duplicate options detected")
            return False
        
        # Check hint exists and is reasonable
        if question.hint and len(question.hint) < 10:
            logging.warning("Hint too short")
            return False
        
        return True
    
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
