import io
import pypdf
import docx
from app.core.supabase import get_supabase
from app.schemas.document import DocumentUpdate
from app.services.analysis_service import AnalysisService
from app.services.quiz_generation_service import QuizGenerationService

class ExtractionService:
    def __init__(self):
        self.supabase = get_supabase()
        self.bucket_name = "course-materials"
        self.analysis_service = AnalysisService()
        self.quiz_service = QuizGenerationService()

    async def process_document(self, document_id: str):
        try:
            # 1. Get document metadata
            doc_response = self.supabase.table("documents").select("*").eq("id", document_id).single().execute()  # type: ignore
            if not doc_response.data:
                print(f"Document {document_id} not found")
                return
            
            doc = doc_response.data
            
            # Type check the document data
            if not isinstance(doc, dict):
                print(f"Invalid document data format for {document_id}")
                return
            
            # 2. Update status to processing
            self._update_status(document_id, "processing")

            # 3. Download file
            file_path = doc.get("file_path")
            if not file_path or not isinstance(file_path, str):
                raise ValueError("Document missing file_path")
                
            file_content = self.supabase.storage.from_(self.bucket_name).download(file_path)
            
            # 4. Extract text
            file_type = doc.get("file_type")
            if not file_type or not isinstance(file_type, str):
                raise ValueError("Document missing file_type")
                
            extracted_text = self._extract_text(file_content, file_type)

            # Save extracted text immediately so it's not lost if analysis fails
            self.supabase.table("documents").update({
                "extracted_text": extracted_text
            }).eq("id", document_id).execute()  # type: ignore

            # 5. Extract Structure (Topics & Concepts)
            print(f"Analyzing content for document {document_id}...")
            await self.analysis_service.analyze_document(document_id, extracted_text)

            # 6. Mark document as completed BEFORE quiz generation
            # This ensures concepts are committed and visible to quiz generation
            self.supabase.table("documents").update({
                "status": "completed", 
                "error_message": None
            }).eq("id", document_id).execute()  # type: ignore
            
            print(f"Successfully processed document {document_id}")

            # 7. Generate Quiz Questions (after document is marked completed)
            print(f"Generating quiz questions for document {document_id}...")
            user_id = doc.get("user_id")
            if user_id and isinstance(user_id, str):
                quiz_id = await self.quiz_service.generate_quiz_for_document(
                    document_id=document_id,
                    user_id=user_id
                )
                if quiz_id:
                    print(f"Successfully generated quiz {quiz_id} for document {document_id}")
                else:
                    print(f"Quiz generation failed for document {document_id}, but document processing completed")
            else:
                print(f"No user_id found for document {document_id}, skipping quiz generation")

        except Exception as e:
            print(f"Error processing document {document_id}: {str(e)}")
            self.supabase.table("documents").update({
                "status": "failed", 
                "error_message": str(e)
            }).eq("id", document_id).execute()  # type: ignore

    def _extract_text(self, file_content: bytes, file_type: str) -> str:
        text = ""
        file_stream = io.BytesIO(file_content)

        if "pdf" in file_type:
            pdf_reader = pypdf.PdfReader(file_stream)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        elif "word" in file_type or "docx" in file_type: # application/vnd.openxmlformats-officedocument.wordprocessingml.document
            doc = docx.Document(file_stream)
            for para in doc.paragraphs:
                text += para.text + "\n"
        
        elif "text" in file_type: # text/plain
            text = file_content.decode('utf-8')
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return text.strip()

    def _update_status(self, document_id: str, status: str):
        self.supabase.table("documents").update({"status": status}).eq("id", document_id).execute()  # type: ignore
