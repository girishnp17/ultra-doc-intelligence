# Ultra Doc-Intelligence

AI-powered logistics document assistant built with RAG. Upload shipping documents, ask natural language questions, and extract structured shipment data — with hallucination guardrails and confidence scoring.

**Live Demo:** [ultra-doc-intelligence.onrender.com](https://ultra-doc-intelligence.onrender.com)

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI |
| LLM | Gemini 2.5 Flash |
| Embeddings | gemini-embedding-001 (3072 dims) |
| Vector Store | ChromaDB (cosine similarity) |
| RAG Framework | LangChain |
| Document Parsing | pdfplumber, python-docx |

## Architecture

```
Client ──▶ FastAPI
              ├── /upload  ──▶ Parse ──▶ Chunk ──▶ Embed ──▶ ChromaDB
              ├── /ask     ──▶ Embed Query ──▶ ChromaDB Top-3 ──▶ Gemini
              └── /extract ──▶ Full Text ──▶ Gemini ──▶ Structured JSON
```

## Project Structure

```
app/
├── main.py              # FastAPI app, static file serving
├── config.py            # Environment variables, model config
├── schemas.py           # Pydantic request/response models
├── routers/
│   └── documents.py     # /upload, /ask, /extract endpoints
└── services/
    ├── parser.py        # PDF, DOCX, TXT document parsing
    ├── rag.py           # Chunking, embedding, vector retrieval
    └── llm.py           # LLM calls for Q&A and extraction
```

## Getting Started

### Prerequisites

- Python 3.12+
- [Google AI API key](https://aistudio.google.com/apikey)

### Run Locally

```bash
git clone https://github.com/girishnp17/ultra-doc-intelligence.git
cd ultra-doc-intelligence
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
uvicorn app.main:app --reload --port 8000
```

### Run with Docker

```bash
docker build -t ultra-doc-intelligence .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-key" ultra-doc-intelligence
```

Open **http://localhost:8000**

## API Reference

| Method | Endpoint | Input | Response |
|---|---|---|---|
| POST | `/upload` | PDF / DOCX / TXT file | `{ doc_id, filename, num_chunks }` |
| POST | `/ask` | `{ doc_id, question }` | `{ answer, source_text, confidence_score }` |
| POST | `/extract` | `{ doc_id }` | Structured shipment JSON |

## How It Works

### Document Processing
Documents are parsed into sections (tables and text blocks separately), then chunked using `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap). Each chunk is embedded and stored in a per-document ChromaDB collection.

### RAG Pipeline
User queries are embedded and matched against stored chunks via cosine similarity. The top-3 chunks plus the full document text are passed to Gemini for grounded answer generation.

### Guardrails
- **Similarity threshold** — Queries with best-chunk similarity below 0.3 are rejected before reaching the LLM
- **Grounded system prompt** — LLM instructed to answer only from provided context; responds "Not found in document" otherwise
- **Dual context verification** — Retrieved chunks and full document text both provided for cross-reference

### Confidence Score
Average cosine similarity of top-3 retrieved chunks, displayed as High (>=65%), Medium (>=45%), or Low (<45%).

### Structured Extraction
Extracts: `shipment_id`, `shipper`, `consignee`, `pickup_datetime`, `delivery_datetime`, `equipment_type`, `mode`, `rate`, `currency`, `weight`, `carrier_name`. Missing fields return `null`.

## Limitations
- Scanned / image-only PDFs require OCR (not currently supported)
- Single-document queries only; cross-document search not supported
- Complex merged-cell tables may not parse perfectly
