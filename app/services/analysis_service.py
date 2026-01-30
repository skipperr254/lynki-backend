import json
import logging
import re
from typing import List, Dict, Any
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from app.core.config import get_settings
from app.core.supabase import get_supabase

settings = get_settings()

class AnalysisService:
    def __init__(self):
        self.supabase = get_supabase()
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = "claude-3-haiku-20240307" # Using Haiku for speed/cost, can upgrade to Sonnet

    async def analyze_document(self, document_id: str, text: str):
        """
        Analyzes the extracted text to identify topics and concepts using Claude.
        Uses chunking to handle large documents and output token limits.
        """
        if not text or len(text) < 50:
            logging.warning(f"Text too short for analysis: Document {document_id}")
            return

        try:
            # Chunk the text (smaller chunks = less output = fits in token limit)
            chunks = self._chunk_text(text, chunk_size=8000)
            print(f"Split document {document_id} into {len(chunks)} chunks for analysis.")

            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}...")
                await self._process_chunk(document_id, chunk, i, len(chunks))

        except Exception as e:
            logging.error(f"Analysis failed for {document_id}: {str(e)}")
            raise e

    def _chunk_text(self, text: str, chunk_size: int = 8000) -> List[str]:
        """Smaller chunks prevent hitting token limits. Character-based for simplicity."""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
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

    async def _process_chunk(self, document_id: str, text_chunk: str, chunk_index: int, total_chunks: int):
        # Retry logic for JSON parsing
        max_retries = 2
        for attempt in range(max_retries + 1):
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
                
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,  # Haiku's safe limit
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.1
                )

                # Type-safe extraction of text content
                content_block = response.content[0]
                if not isinstance(content_block, TextBlock):
                    raise ValueError(f"Unexpected content type: {type(content_block).__name__}")
                
                response_text = content_block.text
                
                # Check if response was truncated
                if response.stop_reason == "max_tokens":
                    logging.warning(f"Chunk {chunk_index+1} hit token limit. Response may be truncated.")
                    # Try to salvage what we have, but flag it
                
                # Clean and extract JSON
                json_str = self._extract_and_clean_json(response_text)
                data = json.loads(json_str)

                # Save to Database (Smart Merge)
                await self._save_structure(document_id, data)

                # Log Usage
                await self._log_usage(document_id, "structure_extraction_chunk", response.usage)
                
                break # Success

            except json.JSONDecodeError as e:
                logging.error(f"Attempt {attempt+1}: Failed to parse JSON from Claude: {e}")
                if attempt < max_retries:
                    logging.error(f"Retrying chunk {chunk_index+1}...")
                else:
                    # Log error with actual output for debugging
                    logging.error(f"Failed to process chunk {chunk_index+1} after {max_retries+1} attempts.")
                    # Only log response_text if it exists (it might not if error occurred before response)
            except Exception as e:
                logging.error(f"Unexpected error processing chunk {chunk_index+1}: {e}")
                raise e

    async def _save_structure(self, document_id: str, data: Dict[str, Any]):
        topics = data.get("topics", [])
        
        for topic_data in topics:
            topic_name = topic_data.get("name")
            if not topic_name:
                logging.warning("Topic missing 'name' field, skipping")
                continue
            
            # 1. Check if topic already exists for this document to avoid duplicates
            existing_topic = self.supabase.table("topics").select("id").eq("document_id", document_id).eq("name", topic_name).execute()  # type: ignore
            
            topic_id = None
            if existing_topic.data and isinstance(existing_topic.data, list) and len(existing_topic.data) > 0:
                # Safely access the first item
                first_topic = existing_topic.data[0]
                if isinstance(first_topic, dict) and "id" in first_topic:
                    topic_id = first_topic["id"]
            
            if not topic_id:
                # Insert New Topic
                topic_res = self.supabase.table("topics").insert({
                    "document_id": document_id,
                    "name": topic_name
                }).execute()  # type: ignore
                
                if topic_res.data and isinstance(topic_res.data, list) and len(topic_res.data) > 0:
                    first_new_topic = topic_res.data[0]
                    if isinstance(first_new_topic, dict) and "id" in first_new_topic:
                        topic_id = first_new_topic["id"]
                
                if not topic_id:
                    logging.warning(f"Failed to insert topic: {topic_name}")
                    continue
            
            # Insert Concepts (always insert, duplicates in concepts might be okay if they have different source_text, or we could check)
            concepts = topic_data.get("concepts", [])
            if not concepts or not isinstance(concepts, list):
                continue
                
            concept_rows = []
            for concept in concepts:
                if not isinstance(concept, dict):
                    continue
                    
                concept_name = concept.get("name")
                if not concept_name:
                    continue
                    
                concept_rows.append({
                    "topic_id": topic_id,
                    "name": concept_name,
                    "explanation": concept.get("explanation", ""),
                    "source_text": concept.get("source_text", ""),
                    "complexity_level": "intermediate"
                })
            
            if concept_rows:
                try:
                    self.supabase.table("concepts").insert(concept_rows).execute()
                except Exception as e:
                    logging.error(f"Failed to insert concepts for topic {topic_name}: {e}")

    async def _log_usage(self, document_id: str, operation: str, usage: Any):
        self.supabase.table("llm_logs").insert({
            "document_id": document_id,
            "operation": operation,
            "model": self.model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens
        }).execute()
