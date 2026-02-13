from pydantic import BaseModel


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    message: str


class AskRequest(BaseModel):
    doc_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    source_text: str | None
    confidence_score: float


class ExtractRequest(BaseModel):
    doc_id: str
