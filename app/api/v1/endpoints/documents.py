from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from app.core.async_db import db_select_single
from app.core.auth import get_current_user_id
from app.core.supabase import get_supabase
from app.schemas.document import DocumentProcessRequest
from app.services.extraction_service import ExtractionService

router = APIRouter()
extraction_service = ExtractionService()
_supabase = get_supabase()

@router.post("/process/{document_id}")
async def process_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    caller: str = Depends(get_current_user_id),
):
    """
    Trigger background processing for a document.
    """
    resp = await db_select_single(_supabase, "documents", "user_id", id=document_id)
    document = getattr(resp, "data", None)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if document["user_id"] != caller:
        raise HTTPException(status_code=403, detail="Not your data")

    background_tasks.add_task(extraction_service.process_document, document_id)
    return {"message": "Document processing started", "document_id": document_id}
