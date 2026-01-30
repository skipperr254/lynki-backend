from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.schemas.document import DocumentProcessRequest
from app.services.extraction_service import ExtractionService

router = APIRouter()
extraction_service = ExtractionService()

@router.post("/process/{document_id}")
async def process_document(
    document_id: str, 
    background_tasks: BackgroundTasks
):
    """
    Trigger background processing for a document.
    """
    background_tasks.add_task(extraction_service.process_document, document_id)
    return {"message": "Document processing started", "document_id": document_id}
