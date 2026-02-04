import asyncio
import io
import logging
import pypdf
import docx
from pptx import Presentation
from app.core.supabase import get_supabase
from app.core.async_db import run_db_operation, db_storage_download
from app.services.analysis_service import AnalysisService
from app.services.quiz_generation_service import QuizGenerationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum time allowed for entire document processing (10 minutes)
DOCUMENT_PROCESSING_TIMEOUT = 600


class ExtractionService:
    def __init__(self):
        self.supabase = get_supabase()
        self.bucket_name = "course-materials"
        self.analysis_service = AnalysisService()
        self.quiz_service = QuizGenerationService()

    async def process_document(self, document_id: str):
        """
        Process a document through the full pipeline:
        1. Download from storage
        2. Extract text
        3. Analyze with Claude (extract topics/concepts)
        4. Generate quiz questions

        All database operations are async to prevent blocking the event loop.
        """
        try:
            # Wrap entire processing in a timeout
            await asyncio.wait_for(
                self._process_document_internal(document_id),
                timeout=DOCUMENT_PROCESSING_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"Document {document_id}: Processing timed out after {DOCUMENT_PROCESSING_TIMEOUT}s")
            await self._update_status_with_error(
                document_id,
                "failed",
                "Processing timed out. The document may be too large or complex. Please try a smaller document."
            )
        except Exception as e:
            logger.exception(f"Document {document_id}: Unexpected error during processing")
            await self._update_status_with_error(
                document_id,
                "failed",
                "An unexpected error occurred during processing. Please try again."
            )

    async def _process_document_internal(self, document_id: str):
        """Internal processing logic with proper async handling."""
        try:
            logger.info(f"Starting processing for document {document_id}")

            # 1. Get document metadata (ASYNC)
            doc_response = await run_db_operation(
                lambda: self.supabase.table("documents").select("*").eq("id", document_id).single().execute()
            )

            if not doc_response.data:
                logger.error(f"Document {document_id} not found in database")
                await self._update_status_with_error(document_id, "failed", "Document not found in database")
                return

            doc = doc_response.data

            # Type check the document data
            if not isinstance(doc, dict):
                logger.error(f"Invalid document data format for {document_id}")
                await self._update_status_with_error(document_id, "failed", "Invalid document data format")
                return

            # 2. Update status to processing (ASYNC)
            await self._update_status(document_id, "processing")
            logger.info(f"Document {document_id}: Status updated to 'processing'")

            # 3. Download file (ASYNC)
            file_path = doc.get("file_path")
            if not file_path or not isinstance(file_path, str):
                raise ValueError("Document is missing file path")

            logger.info(f"Document {document_id}: Downloading file from storage...")
            try:
                file_content = await db_storage_download(self.supabase, self.bucket_name, file_path)
            except Exception as e:
                raise ValueError(f"Failed to download file from storage: {str(e)}")

            # 4. Extract text (CPU-bound, run in executor to not block)
            file_type = doc.get("file_type")
            if not file_type or not isinstance(file_type, str):
                raise ValueError("Document is missing file type")

            logger.info(f"Document {document_id}: Extracting text from {file_type}...")
            try:
                # Run CPU-bound text extraction in executor
                loop = asyncio.get_event_loop()
                extracted_text = await loop.run_in_executor(
                    None,
                    lambda: self._extract_text(file_content, file_type)
                )
            except ValueError as e:
                raise ValueError(f"Text extraction failed: {str(e)}")

            if not extracted_text or len(extracted_text.strip()) < 50:
                raise ValueError("Extracted text is too short or empty. Please upload a document with more content.")

            # Save extracted text immediately so it's not lost if analysis fails (ASYNC)
            await run_db_operation(
                lambda: self.supabase.table("documents").update({
                    "extracted_text": extracted_text
                }).eq("id", document_id).execute()
            )
            logger.info(f"Document {document_id}: Extracted {len(extracted_text)} characters of text")

            # 5. Extract Structure (Topics & Concepts)
            logger.info(f"Document {document_id}: Starting AI analysis...")
            try:
                await self.analysis_service.analyze_document(document_id, extracted_text)
            except Exception as e:
                logger.error(f"Document {document_id}: Analysis failed - {str(e)}")
                raise ValueError(f"AI analysis failed: {str(e)}")

            # Verify concepts were created (ASYNC)
            concepts_count = await self._count_document_concepts(document_id)
            if concepts_count == 0:
                raise ValueError("No concepts could be extracted from the document. The content may not be suitable for quiz generation.")

            logger.info(f"Document {document_id}: Analysis complete - {concepts_count} concepts extracted")

            # 6. Mark document as completed BEFORE quiz generation (ASYNC)
            await run_db_operation(
                lambda: self.supabase.table("documents").update({
                    "status": "completed",
                    "error_message": None
                }).eq("id", document_id).execute()
            )
            logger.info(f"Document {document_id}: Status updated to 'completed'")

            # 7. Generate Quiz Questions (after document is marked completed)
            logger.info(f"Document {document_id}: Starting quiz generation...")
            user_id = doc.get("user_id")
            if user_id and isinstance(user_id, str):
                quiz_id = await self.quiz_service.generate_quiz_for_document(
                    document_id=document_id,
                    user_id=user_id
                )
                if quiz_id:
                    logger.info(f"Document {document_id}: Quiz {quiz_id} generated successfully")
                else:
                    logger.warning(f"Document {document_id}: Quiz generation failed, but document processing completed")
            else:
                logger.warning(f"Document {document_id}: No user_id found, skipping quiz generation")

        except ValueError as e:
            # User-friendly errors (validation, unsupported file types, etc.)
            error_message = str(e)
            logger.error(f"Document {document_id}: Processing failed - {error_message}")
            await self._update_status_with_error(document_id, "failed", error_message)

    async def _count_document_concepts(self, document_id: str) -> int:
        """Count the number of concepts extracted for a document (ASYNC)."""
        try:
            # Get topics for document
            topics_response = await run_db_operation(
                lambda: self.supabase.table("topics").select("id").eq("document_id", document_id).execute()
            )
            if not topics_response.data:
                return 0

            topic_ids = [t["id"] for t in topics_response.data]

            # Count concepts for those topics
            concepts_response = await run_db_operation(
                lambda: self.supabase.table("concepts").select("id", count="exact").in_("topic_id", topic_ids).execute()
            )
            return concepts_response.count if concepts_response.count else 0
        except Exception:
            return 0

    async def _update_status_with_error(self, document_id: str, status: str, error_message: str):
        """Update document status and error message (ASYNC)."""
        try:
            await run_db_operation(
                lambda: self.supabase.table("documents").update({
                    "status": status,
                    "error_message": error_message
                }).eq("id", document_id).execute()
            )
        except Exception as e:
            logger.error(f"Failed to update document {document_id} status: {e}")

    async def _update_status(self, document_id: str, status: str):
        """Update document status (ASYNC)."""
        await run_db_operation(
            lambda: self.supabase.table("documents").update({"status": status}).eq("id", document_id).execute()
        )

    def _extract_text(self, file_content: bytes, file_type: str) -> str:
        """
        Extract text from file content. This is CPU-bound and should be run in an executor.
        """
        text = ""
        file_stream = io.BytesIO(file_content)

        if "pdf" in file_type:
            pdf_reader = pypdf.PdfReader(file_stream)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

        elif "word" in file_type or "docx" in file_type:
            # application/vnd.openxmlformats-officedocument.wordprocessingml.document
            doc = docx.Document(file_stream)

            # Extract from paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"

            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if cell_text:
                            row_text.append(cell_text)
                    if row_text:
                        text += " | ".join(row_text) + "\n"

        elif "powerpoint" in file_type or "pptx" in file_type or "presentation" in file_type:
            # application/vnd.openxmlformats-officedocument.presentationml.presentation
            prs = Presentation(file_stream)

            for slide_num, slide in enumerate(prs.slides, 1):
                slide_text = []

                for shape in slide.shapes:
                    # Extract text from text frames
                    if shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            para_text = ""
                            for run in paragraph.runs:
                                if run.text:
                                    para_text += run.text
                            if para_text.strip():
                                slide_text.append(para_text.strip())

                    # Extract text from tables in slides
                    if shape.has_table:
                        for row in shape.table.rows:
                            row_text = []
                            for cell in row.cells:
                                if cell.text.strip():
                                    row_text.append(cell.text.strip())
                            if row_text:
                                slide_text.append(" | ".join(row_text))

                if slide_text:
                    text += f"--- Slide {slide_num} ---\n"
                    text += "\n".join(slide_text) + "\n\n"

        elif "text" in file_type:  # text/plain
            text = file_content.decode('utf-8')

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return text.strip()
