import json
import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from anthropic import AsyncAnthropic, APITimeoutError, APIConnectionError
from anthropic.types import TextBlock
from app.core.config import get_settings
from app.core.supabase import get_supabase
from app.core.async_db import run_db_operation

settings = get_settings()

# Timeout configuration
CLAUDE_TIMEOUT_SECONDS = 60  # 60 seconds for Haiku analysis
MAX_API_RETRIES = 2
# Bounds concurrent Claude chunk-analysis calls per document. Sized to cover
# a typical document in a single wave: at chunk_size=8000 a ~60k-char chapter
# splits into ~10 chunks, and a bound of 5 made that two waves — measured at
# 42.0s vs 28.6s for one wave, with identical extraction yield. This is a
# per-document bound, so N documents processing at once put N times this many
# calls in flight; if that starts drawing 429s, add a process-wide gate rather
# than lowering this back down.
CHUNK_ANALYSIS_MAX_CONCURRENCY = 10

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self):
        self.supabase = get_supabase()
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = "claude-sonnet-4-6"  # Using Sonnet

    async def analyze_document(self, document_id: str, text: str):
        """
        Analyzes the extracted text to identify topics and concepts using Claude.
        Uses chunking to handle large documents and output token limits.
        All database operations are async to prevent blocking.
        """
        if not text or len(text) < 50:
            logger.warning(f"Text too short for analysis: Document {document_id}")
            return

        try:
            # Chunk the text (smaller chunks = less output = fits in token limit)
            chunks = self._chunk_text(text, chunk_size=8000)
            logger.info(f"Split document {document_id} into {len(chunks)} chunks for analysis.")

            # Phase 1: run all chunk API calls concurrently (bounded, no shared
            # state — each call just returns parsed data, nothing is saved yet).
            semaphore = asyncio.Semaphore(CHUNK_ANALYSIS_MAX_CONCURRENCY)
            tasks = [
                asyncio.create_task(
                    self._analyze_chunk(document_id, chunk, i, len(chunks), semaphore)
                )
                for i, chunk in enumerate(chunks)
            ]

            try:
                results = await asyncio.gather(*tasks)
            except Exception:
                # An unexpected error in one chunk aborts the whole document,
                # same as before. Cancel siblings so we don't leave orphaned
                # Claude calls running in the background after we've already
                # decided this run failed.
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

            # Phase 2: merge every chunk's topics in memory (chunk order is
            # preserved, so cross-chunk topic dedup resolves exactly as the
            # old per-chunk read-then-write did), then save the whole
            # document at once. Saving per chunk cost 2 sequential round
            # trips per topic — ~54 on a 10-chunk document — which is pure
            # network latency and the dominant cost of this method now that
            # the API calls run in a single wave.
            merged: Dict[str, List[Dict[str, Any]]] = {}
            usages: List[Any] = []
            for i, result in enumerate(results):
                if result is None:
                    continue  # chunk exhausted retries; nothing to save
                data, usage = result
                self._merge_structure(merged, data)
                usages.append(usage)
                logger.info(f"Chunk {i+1}/{len(chunks)} merged successfully")

            await self._save_structure(document_id, merged)
            await self._log_usage(document_id, "structure_extraction_chunk", usages)

        except Exception as e:
            logger.error(f"Analysis failed for {document_id}: {str(e)}")
            raise e

    def _chunk_text(self, text: str, chunk_size: int = 8000) -> List[str]:
        """
        Split text into chunks that respect paragraph boundaries.
        This prevents cutting content mid-sentence and preserves context.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        current_chunk = ""

        # Split by double newlines (paragraphs) first, then single newlines
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph exceeds the limit
            if len(current_chunk) + len(paragraph) + 2 > chunk_size:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # If a single paragraph is too long, split by sentences
                if len(paragraph) > chunk_size:
                    sentences = self._split_into_sentences(paragraph)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 > chunk_size:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            current_chunk += (" " if current_chunk else "") + sentence
                else:
                    current_chunk = paragraph
            else:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for finer granularity."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_and_clean_json(self, text: str) -> str:
        """Extract JSON from response and clean common formatting issues."""
        # Remove markdown code blocks if present
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)

        # Find JSON object boundaries
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")

        json_str = text[start_idx:end_idx]

        # Fix common JSON issues
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        return json_str

    async def _analyze_chunk(
        self,
        document_id: str,
        text_chunk: str,
        chunk_index: int,
        total_chunks: int,
        semaphore: asyncio.Semaphore,
    ) -> Optional[Tuple[Dict[str, Any], Any]]:
        """
        Call Claude + parse JSON for a single chunk, with retry logic.
        API-call phase only — does NOT save to the database. Returns
        (data, usage) on success, or None if retries were exhausted
        (best-effort; caller skips saving for this chunk). A genuinely
        unexpected error still raises, aborting the whole document.
        """
        async with semaphore:
            for attempt in range(MAX_API_RETRIES + 1):
                try:
                    system_prompt = (
                        "You are an expert educational curriculum designer. "
                        "Analyze this course material section and extract key learning elements.\n\n"
                        "Identify main Topics. For each Topic, list the key Concepts. "
                        "For each Concept: provide a concise explanation (1-2 sentences) and extract a relevant quote as source_text.\n\n"
                        "CRITICAL JSON RULES:\n"
                        "- Output ONLY valid JSON\n"
                        "- NO markdown code blocks\n"
                        "- NO trailing commas\n"
                        "- Keep explanations concise (under 100 words each)\n"
                        "- Limit to 5-10 concepts per topic maximum\n\n"
                        "Format: {{\"topics\": [{{\"name\": \"Topic\", \"concepts\": [{{\"name\": \"Concept\", \"explanation\": \"Brief explanation\", \"source_text\": \"Quote\"}}]}}]}}\n\n"
                        "Example: {{\"topics\": [{{\"name\": \"Machine Learning\", \"concepts\": [{{\"name\": \"Neural Networks\", \"explanation\": \"Computational models inspired by brain structure\", \"source_text\": \"Neural networks consist of interconnected nodes...\"}}]}}]}}"
                    )

                    user_message = f"Content (Chunk {chunk_index+1}/{total_chunks}):\n\n{text_chunk}"

                    # Use asyncio.wait_for for timeout handling
                    response = await asyncio.wait_for(
                        self.client.messages.create(
                            model=self.model,
                            max_tokens=4000,  # Haiku's safe limit
                            system=system_prompt,
                            messages=[
                                {"role": "user", "content": user_message}
                            ],
                            temperature=0.1
                        ),
                        timeout=CLAUDE_TIMEOUT_SECONDS
                    )

                    # Type-safe extraction of text content
                    content_block = response.content[0]
                    if not isinstance(content_block, TextBlock):
                        raise ValueError(f"Unexpected content type: {type(content_block).__name__}")

                    response_text = content_block.text

                    # Check if response was truncated
                    if response.stop_reason == "max_tokens":
                        logger.warning(f"Chunk {chunk_index+1} hit token limit. Response may be truncated.")

                    # Clean and extract JSON
                    json_str = self._extract_and_clean_json(response_text)
                    data = json.loads(json_str)

                    logger.info(f"Chunk {chunk_index+1}/{total_chunks} processed successfully")
                    return data, response.usage  # Success

                except asyncio.TimeoutError:
                    logger.error(f"Attempt {attempt+1}: Claude API timeout after {CLAUDE_TIMEOUT_SECONDS}s for chunk {chunk_index+1}")
                    if attempt < MAX_API_RETRIES:
                        logger.info(f"Retrying chunk {chunk_index+1} after timeout...")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    else:
                        logger.error(f"Failed to process chunk {chunk_index+1} after {MAX_API_RETRIES+1} attempts due to timeouts")
                        # Don't raise - continue with other chunks

                except (APITimeoutError, APIConnectionError) as e:
                    logger.error(f"Attempt {attempt+1}: Claude API connection error for chunk {chunk_index+1}: {e}")
                    if attempt < MAX_API_RETRIES:
                        logger.info(f"Retrying chunk {chunk_index+1} after connection error...")
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"Failed to process chunk {chunk_index+1} after {MAX_API_RETRIES+1} attempts")

                except json.JSONDecodeError as e:
                    logger.error(f"Attempt {attempt+1}: Failed to parse JSON from Claude: {e}")
                    if attempt < MAX_API_RETRIES:
                        logger.info(f"Retrying chunk {chunk_index+1} due to JSON error...")
                    else:
                        logger.error(f"Failed to process chunk {chunk_index+1} after {MAX_API_RETRIES+1} attempts.")

                except Exception as e:
                    logger.error(f"Unexpected error processing chunk {chunk_index+1}: {e}")
                    raise e

            return None  # retries exhausted (best-effort; caller skips saving)

    @staticmethod
    def _merge_structure(
        merged: Dict[str, List[Dict[str, Any]]],
        data: Dict[str, Any],
    ) -> None:
        """Fold one chunk's parsed topics into the document-wide structure.

        Topic name is the dedup key, exactly as the old per-chunk
        SELECT-by-name was: a topic named by several chunks collects all of
        their concepts under one entry. Pure and in-memory — chunk iteration
        order is the only thing deciding the final ordering.
        """
        for topic_data in data.get("topics", []) or []:
            if not isinstance(topic_data, dict):
                continue

            topic_name = topic_data.get("name")
            if not topic_name:
                logger.warning("Topic missing 'name' field, skipping")
                continue

            # setdefault, not a guarded insert: a topic with no usable
            # concepts still gets its row, as it did before.
            bucket = merged.setdefault(topic_name, [])

            concepts = topic_data.get("concepts")
            if not concepts or not isinstance(concepts, list):
                continue

            for concept in concepts:
                if not isinstance(concept, dict):
                    continue

                concept_name = concept.get("name")
                if not concept_name:
                    continue

                bucket.append({
                    "name": concept_name,
                    "explanation": concept.get("explanation", ""),
                    "source_text": concept.get("source_text", ""),
                    "complexity_level": "intermediate",
                })

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

        # Every topic for the document, rather than an `in_` over the names —
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
            # position — PostgREST promises no ordering for it.
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
        """Log every chunk's API usage in one insert (ASYNC)."""
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
