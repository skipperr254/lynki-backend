import logging
import asyncio
from typing import List, Dict, Any, Tuple
from anthropic import AsyncAnthropic, APITimeoutError, APIConnectionError
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from app.core.config import get_settings
from app.core.supabase import get_supabase
from app.core.async_db import run_db_operation

settings = get_settings()

# One analysis call now covers a whole document rather than an ~8000-char
# chunk, so it emits far more output tokens and runs serially instead of as a
# parallel wave -- minutes, not the ~30s a chunk wave took. 300s covers that
# with room to spare and still sits well inside extraction_service's 600s
# DOCUMENT_PROCESSING_TIMEOUT, which has to fit OCR in the same budget.
ANALYSIS_TIMEOUT_SECONDS = 300
MAX_API_RETRIES = 2

# Refuse oversized documents rather than truncate or re-introduce splitting.
# The model's context window is ~4M characters, so this cap leaves a wide
# margin for the output budget and prompt overhead while still sitting far
# above any realistic upload (a dense 300-page chapter is well under 1M).
MAX_ANALYSIS_CHARS = 1_500_000

# Bounds the response so a pathological document cannot ask for an unbounded
# one. The model allows up to 128k output tokens; 16k is on the order of 200
# concepts, more than any single document should legitimately yield.
MAX_OUTPUT_TOKENS = 16000

logger = logging.getLogger(__name__)


class ConceptOut(BaseModel):
    """One extracted concept.

    `extra="forbid"` is what puts `additionalProperties: false` into the
    generated JSON schema, which structured outputs requires.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The concept as a student would refer to it.")
    explanation: str = Field(
        description="1-2 sentences on what this concept actually says."
    )
    source_text: str = Field(
        description="A short verbatim quote from the material that grounds this concept."
    )


class TopicOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The topic heading.")
    concepts: List[ConceptOut] = Field(
        description="The key concepts taught under this topic."
    )


class DocumentStructure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topics: List[TopicOut] = Field(
        description="Topics in the order they appear in the material."
    )


class AnalysisService:
    def __init__(self):
        self.supabase = get_supabase()
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = "claude-sonnet-5"

    async def analyze_document(self, document_id: str, text: str):
        """
        Analyze a document's full extracted text in a single Claude call and
        save the resulting topics/concepts.

        The text is sent whole rather than in ~8000-char chunks. Chunking
        existed only to stay inside a context limit that the current model's
        1M-token window makes irrelevant, and it was actively splitting single
        concepts across two calls -- producing two half-grounded concept rows
        for one idea, each of which then became its own BKT knowledge
        component. All database operations are async to avoid blocking the
        event loop.
        """
        if not text or len(text) < 50:
            logger.warning(f"Text too short for analysis: Document {document_id}")
            return

        if len(text) > MAX_ANALYSIS_CHARS:
            # User-facing: extraction_service re-raises ValueError unchanged.
            raise ValueError(
                "This document is too large for us to analyse in one pass. "
                "Try uploading it a chapter at a time."
            )

        try:
            logger.info(
                f"Analyzing document {document_id} as a whole "
                f"({len(text)} characters)"
            )
            structure, usage = await self._extract_structure(text)

            merged = self._to_save_shape(structure)
            await self._save_structure(document_id, merged)
            await self._log_usage(document_id, "structure_extraction", [usage])

        except Exception as e:
            logger.error(f"Analysis failed for {document_id}: {str(e)}")
            raise e

    async def _extract_structure(self, text: str) -> Tuple[DocumentStructure, Any]:
        """Call Claude once for the whole document and return the validated
        structure plus its usage.

        Uses structured outputs (`messages.parse` with a Pydantic
        `output_format`) rather than free-text JSON. The schema is enforced
        server-side, so the markdown-fence-stripping and trailing-comma-
        rewriting repair path this replaced has nothing left to repair --
        which matters far more now that one malformed response would cost the
        whole document instead of one chunk of it.

        Retries cover transport failures only. A `ValueError` raised in here
        is terminal and user-facing, so it deliberately escapes the loop.
        """
        system_prompt = self._build_system_prompt(text)
        last_error: Exception | None = None

        for attempt in range(MAX_API_RETRIES + 1):
            try:
                response = await asyncio.wait_for(
                    self.client.messages.parse(
                        model=self.model,
                        max_tokens=MAX_OUTPUT_TOKENS,
                        system=system_prompt,
                        messages=[{"role": "user", "content": text}],
                        output_format=DocumentStructure,
                    ),
                    timeout=ANALYSIS_TIMEOUT_SECONDS,
                )

                if response.stop_reason == "max_tokens":
                    # Truncated output used to be logged as a warning and then
                    # parsed anyway, silently dropping everything past the cut.
                    # With one call per document that would quietly cost the
                    # back half of the material, so fail loudly instead.
                    raise ValueError(
                        "This document produced more material than we can analyse "
                        "in one pass. Try uploading it a chapter at a time."
                    )

                structure = response.parsed_output
                if structure is None:
                    raise ValueError(
                        "We had trouble reading this document's content. "
                        "Please try again."
                    )

                logger.info(
                    f"Extracted {len(structure.topics)} topic(s), "
                    f"{sum(len(t.concepts) for t in structure.topics)} concept(s)"
                )
                return structure, response.usage

            except (
                asyncio.TimeoutError,
                APITimeoutError,
                APIConnectionError,
                ValidationError,
            ) as e:
                last_error = e
                logger.error(
                    f"Attempt {attempt + 1}/{MAX_API_RETRIES + 1} of document "
                    f"analysis failed: {type(e).__name__}: {e}"
                )
                if attempt < MAX_API_RETRIES:
                    await asyncio.sleep(2 ** attempt)  # 1s, then 2s

        raise RuntimeError(
            f"Document analysis failed after {MAX_API_RETRIES + 1} attempts"
        ) from last_error

    @staticmethod
    def _build_system_prompt(text: str) -> str:
        """System prompt for whole-document analysis.

        Carries no JSON formatting rules at all: the response schema is
        enforced by structured outputs, so the prompt only has to describe the
        work. What it does have to do is push for coverage. Chunking used to
        guarantee that structurally -- every section got its own API call, so
        the model could not skip page 47 -- and a single call over the whole
        document will otherwise drift toward summarising the opening.
        """
        # Roughly one concept per 1200 characters, clamped. Gives a
        # length-aware target instead of the flat "5-10 concepts per topic"
        # that asked the same of a 60k-char chapter and a 6k-char handout.
        target = max(8, min(120, len(text) // 1200))

        return (
            "You are an expert educational curriculum designer. You are given the "
            "COMPLETE text of one piece of course material. Extract the learning "
            "structure a student needs in order to master it.\n\n"
            "Identify the main Topics, and under each Topic the key Concepts. "
            "For each Concept give its name as a student would refer to it, a "
            "1-2 sentence explanation of what it actually says, and a short "
            "verbatim quote from the material that grounds it.\n\n"
            "COVERAGE REQUIREMENTS -- these matter more than brevity:\n"
            "- Work through the material from its beginning all the way to its "
            "end. The closing sections must be represented as thoroughly as the "
            "opening ones.\n"
            "- Emit topics in the order they appear in the material.\n"
            "- Do not summarise and do not stop early. This is an extraction "
            "task, not a synopsis.\n"
            f"- Aim for roughly {target} concepts in total across all topics, "
            "scaled to how much genuinely distinct material is present.\n"
            "- Every concept must be distinct. Never emit the same idea twice "
            "under two different names."
        )

    @staticmethod
    def _to_save_shape(
        structure: DocumentStructure,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fold the model's response into the {topic_name: [concept_row]} shape
        `_save_structure` consumes.

        Topic name remains the dedup key, exactly as it was when chunk results
        were merged: a name the model emits twice collects both entries'
        concepts under a single topic row.
        """
        merged: Dict[str, List[Dict[str, Any]]] = {}

        for topic in structure.topics:
            topic_name = topic.name.strip()
            if not topic_name:
                logger.warning("Topic with empty name, skipping")
                continue

            # setdefault, not a guarded insert: a topic with no usable concepts
            # still gets its row, as it did before.
            bucket = merged.setdefault(topic_name, [])

            for concept in topic.concepts:
                concept_name = concept.name.strip()
                if not concept_name:
                    continue

                bucket.append({
                    "name": concept_name,
                    "explanation": concept.explanation,
                    "source_text": concept.source_text,
                    "complexity_level": "intermediate",
                })

        return merged

    async def _save_structure(
        self,
        document_id: str,
        merged: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Save the merged topic/concept structure in a fixed 3 round trips.

        (1) one SELECT of this document's existing topics, (2) one bulk
        INSERT of the ones it doesn't have yet, (3) one bulk INSERT of every
        concept across every topic. The count is fixed regardless of how many
        topics the document has.

        The existing-topic read still earns its trip: a document reprocessed
        after a partial failure already has rows, and reusing them is what
        keeps a retry from duplicating every topic.
        """
        if not merged:
            return

        topic_names = list(merged.keys())

        # Every topic for the document, rather than an `in_` over the names --
        # the count is small, it costs the same single trip, and it sidesteps
        # quoting topic names that contain commas into a PostgREST filter.
        existing_resp = await run_db_operation(
            lambda: self.supabase.table("topics")
            .select("id, name")
            .eq("document_id", document_id)
            .execute()
        )
        existing = getattr(existing_resp, "data", None) or []
        topic_ids: Dict[str, str] = {
            r["name"]: r["id"]
            for r in existing
            if isinstance(r, dict) and r.get("name") and r.get("id")
        }

        new_names = [n for n in topic_names if n not in topic_ids]
        if new_names:
            new_rows = [{"document_id": document_id, "name": n} for n in new_names]
            insert_resp = await run_db_operation(
                lambda: self.supabase.table("topics").insert(new_rows).execute()
            )
            # Map the returned representation by name rather than by
            # position -- PostgREST promises no ordering for it.
            for r in (getattr(insert_resp, "data", None) or []):
                if isinstance(r, dict) and r.get("name") and r.get("id"):
                    topic_ids[r["name"]] = r["id"]

        unsaved = [n for n in topic_names if n not in topic_ids]
        if unsaved:
            logger.warning(f"Failed to insert topic(s): {', '.join(unsaved)}")

        concept_rows = [
            {**concept, "topic_id": topic_ids[name]}
            for name in topic_names
            if name in topic_ids
            for concept in merged[name]
        ]
        if not concept_rows:
            return

        try:
            await run_db_operation(
                lambda: self.supabase.table("concepts").insert(concept_rows).execute()
            )
        except Exception as e:
            # A single unwritable row would otherwise cost the document every
            # concept it extracted. Fall back to the pre-batching granularity
            # so the blast radius stays one topic.
            logger.error(f"Bulk concept insert failed, retrying per topic: {e}")
            await self._save_concepts_per_topic(merged, topic_ids)

    async def _save_concepts_per_topic(
        self,
        merged: Dict[str, List[Dict[str, Any]]],
        topic_ids: Dict[str, str],
    ) -> None:
        """Fallback path for a rejected bulk concept insert: one INSERT per
        topic, so one bad row loses that topic's concepts and not the rest."""
        for name, concepts in merged.items():
            topic_id = topic_ids.get(name)
            if not topic_id or not concepts:
                continue

            rows = [{**c, "topic_id": topic_id} for c in concepts]
            try:
                await run_db_operation(
                    lambda r=rows: self.supabase.table("concepts").insert(r).execute()
                )
            except Exception as e:
                logger.error(f"Failed to insert concepts for topic {name}: {e}")

    async def _log_usage(self, document_id: str, operation: str, usages: List[Any]):
        """Log the analysis call's API usage in one insert (ASYNC)."""
        rows = [
            {
                "document_id": document_id,
                "operation": operation,
                "model": self.model,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
            }
            for u in usages
        ]
        if not rows:
            return

        try:
            await run_db_operation(
                lambda: self.supabase.table("llm_logs").insert(rows).execute()
            )
        except Exception as e:
            # Don't fail the whole process if logging fails
            logger.warning(f"Failed to log LLM usage: {e}")
