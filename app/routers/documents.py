import os
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import UPLOAD_DIR
from app.schemas import AskRequest, AskResponse, ExtractRequest, UploadResponse
from app.services.parser import parse_document
from app.services.rag import store_chunks, retrieve, get_full_text
from app.services.llm import ask_question, extract_structured

router = APIRouter()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
SIMILARITY_THRESHOLD = 0.3


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """Upload a document (PDF, DOCX, or TXT), parse, chunk, embed, and index."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: PDF, DOCX, TXT.",
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    sections = parse_document(dest_path)
    if not sections:
        raise HTTPException(status_code=422, detail="No text could be extracted from the document")

    doc_id, num_chunks = store_chunks(sections, file.filename)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        num_chunks=num_chunks,
        message="Document uploaded and indexed successfully.",
    )


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Ask a natural-language question about an uploaded document."""
    try:
        chunks = retrieve(req.doc_id, req.question)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_id}' not found")

    best_score = max(c["score"] for c in chunks) if chunks else 0.0
    if best_score < SIMILARITY_THRESHOLD:
        return AskResponse(
            answer="Not found in document.",
            source_text=None,
            confidence_score=round(best_score, 4),
        )

    full_text = get_full_text(req.doc_id)
    result = ask_question(req.question, chunks, full_text)
    return AskResponse(**result)


@router.post("/extract")
async def extract(req: ExtractRequest):
    """Extract structured shipment data from an uploaded document."""
    try:
        full_text = get_full_text(req.doc_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_id}' not found")

    return extract_structured(full_text)
