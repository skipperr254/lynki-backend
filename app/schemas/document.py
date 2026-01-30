from pydantic import BaseModel
from typing import Optional, Literal

class DocumentProcessRequest(BaseModel):
    document_id: str

class DocumentUpdate(BaseModel):
    status: Literal['pending', 'processing', 'completed', 'failed']
    extracted_text: Optional[str] = None
    error_message: Optional[str] = None
