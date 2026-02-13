# Ultra Doc-Intelligence

AI-powered logistics document assistant. Upload a document (BOL, Rate Confirmation, Invoice), ask natural language questions, and extract structured shipment data — grounded in the document with confidence scoring and hallucination guardrails.

**Live Demo:** [https://ultra-doc-intelligence.onrender.com](https://ultra-doc-intelligence.onrender.com)

## Architecture

```
  Browser ──▶ FastAPI Backend
                 │
                 ├── POST /upload  ──▶ Parser ──▶ Chunker ──▶ Embed ──▶ ChromaDB
                 ├── POST /ask     ──▶ ChromaDB Retrieval ──▶ Gemini LLM
                 └── POST /extract ──▶ Full Text ──▶ Gemini LLM
                                            │
                               ┌────────────┴────────────┐
                               │  Gemini 2.5 Flash (LLM) │
                               │  gemini-embedding-001    │
                               │  ChromaDB (vector store) │
                               └─────────────────────────┘
```

**Stack:** FastAPI, LangChain, Gemini 2.5 Flash, ChromaDB, pdfplumber, python-docx

## Project Structure

```
app/
├── main.py                # FastAPI app + static UI
├── config.py              # Environment settings
├── schemas.py             # Pydantic request/response models
├── routers/
│   └── documents.py       # /upload, /ask, /extract endpoints
└── services/
    ├── parser.py           # PDF, DOCX, TXT parsing
    ├── rag.py              # Chunking + embedding + ChromaDB retrieval
    └── llm.py              # Gemini LLM calls (ask + extract)
```

## Setup

```bash
git clone https://github.com/girishnp17/ultra-doc-intelligence.git
cd ultra-doc-intelligence
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
uvicorn app.main:app --reload --port 8000
```

Or with Docker:

```bash
docker build -t ultra-doc-intelligence .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-key" ultra-doc-intelligence
```

Open **http://localhost:8000**

## API Endpoints

| Endpoint | Input | Output |
|---|---|---|
| `POST /upload` | File (PDF/DOCX/TXT) | `{ doc_id, filename, num_chunks }` |
| `POST /ask` | `{ doc_id, question }` | `{ answer, source_text, confidence_score }` |
| `POST /extract` | `{ doc_id }` | JSON with shipment fields (nulls if missing) |

## Technical Approach

### Parsing
- **PDF:** pdfplumber — tables via `extract_tables()` + free text, with deduplication
- **DOCX:** python-docx — tables and paragraph blocks separately
- **TXT:** Splits on double newlines into logical sections

### Chunking
LangChain `RecursiveCharacterTextSplitter` — 1000 char chunks, 200 char overlap. Splits at the most semantically meaningful boundary (`\n\n` > `\n` > `. ` > ` `). Each chunk retains metadata (doc_id, filename, chunk_index).

### Retrieval
Embeddings via `gemini-embedding-001` (3072 dims) stored in ChromaDB with cosine similarity. Queries retrieve top-3 matching chunks with similarity scores.

### Guardrails
1. **Retrieval threshold** — If best chunk cosine similarity < 0.3, returns "Not found in document" without calling the LLM
2. **System prompt grounding** — LLM strictly instructed to only answer from provided context
3. **Dual context** — Both retrieved chunks and full document text passed to LLM for verification

### Confidence Scoring
Average cosine similarity of the top-3 retrieved chunks (0–1 scale). Displayed as High (>=65%), Medium (>=45%), or Low (<45%). Based on retrieval similarity rather than LLM self-assessment — an independent, objective measure.

### Structured Extraction
Gemini extracts: `shipment_id`, `shipper`, `consignee`, `pickup_datetime`, `delivery_datetime`, `equipment_type`, `mode`, `rate`, `currency`, `weight`, `carrier_name`. Returns `null` for missing fields.

## Known Limitations
- **Scanned/image PDFs** — pdfplumber only extracts text-layer; OCR would be needed
- **Very large documents** — full text is sent to LLM; may exceed context window for extremely large docs
- **Multi-document queries** — each document is isolated; cross-document questions not supported
- **Complex tables** — nested or merged-cell tables may not parse perfectly

