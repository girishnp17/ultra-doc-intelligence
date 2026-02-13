# Ultra Doc-Intelligence

AI-powered logistics document assistant. Upload a logistics document (BOL, Rate Confirmation, Invoice), ask natural language questions, and extract structured shipment data — all grounded in the document with confidence scoring and hallucination guardrails.

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│   Browser    │     │              FastAPI Backend                 │
│  (static UI) │────▶│                                              │
└─────────────┘     │  POST /upload ──▶ Parser ──▶ Chunker ──▶ ChromaDB
                    │  POST /ask    ──▶ Retriever ──▶ Gemini LLM   │
                    │  POST /extract──▶ Full Text ──▶ Gemini LLM   │
                    └──────────────────────────────────────────────┘
                              │                │
                    ┌─────────┴───┐   ┌────────┴────────┐
                    │  ChromaDB   │   │  Gemini 2.5     │
                    │ (embedded)  │   │  Flash + Embed  │
                    └─────────────┘   └─────────────────┘
```

**Stack:** FastAPI, LangChain, Gemini 2.5 Flash, ChromaDB, pdfplumber, python-docx

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI app, serves UI
│   ├── config.py            # Environment settings
│   ├── schemas.py           # Pydantic request/response models
│   ├── routers/
│   │   └── documents.py     # /upload, /ask, /extract endpoints
│   └── services/
│       ├── parser.py        # PDF, DOCX, TXT parsing
│       ├── rag.py           # Chunking, embedding, ChromaDB retrieval
│       └── llm.py           # Gemini LLM calls (ask + extract)
├── static/
│   └── index.html           # Chat UI
├── sample_docs/             # Test logistics documents
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Setup & Run

### Prerequisites
- Python 3.12+
- A Google AI API key ([get one here](https://aistudio.google.com/apikey))

### Local Setup

```bash
git clone <repo-url>
cd ultra-doc-intelligence

pip install -r requirements.txt

export GEMINI_API_KEY="your-key-here"
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

### Docker

```bash
docker build -t ultra-doc-intelligence .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-key" ultra-doc-intelligence
```

## API Endpoints

### `POST /upload`
Upload a document (PDF, DOCX, or TXT).

```bash
curl -X POST http://localhost:8000/upload -F "file=@document.pdf"
```

**Response:**
```json
{
  "doc_id": "3a3e35c244eb",
  "filename": "document.pdf",
  "num_chunks": 9,
  "message": "Document uploaded and indexed successfully."
}
```

### `POST /ask`
Ask a question about an uploaded document.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "3a3e35c244eb", "question": "What is the shipment weight?"}'
```

**Response:**
```json
{
  "answer": "The weight of the shipment is 56000 lbs.",
  "source_text": "[Page 1 – Table]\n# Of Units | Weight\n#10000 | 56000 lbs",
  "confidence_score": 0.7056
}
```

### `POST /extract`
Extract structured shipment data as JSON.

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "3a3e35c244eb"}'
```

**Response:**
```json
{
  "shipment_id": "LD53657",
  "shipper": "AAA",
  "consignee": "xyz",
  "pickup_datetime": "02-08-2026 09:00",
  "delivery_datetime": "02-08-2026 09:00",
  "equipment_type": null,
  "mode": null,
  "rate": null,
  "currency": null,
  "weight": "56000 lbs",
  "carrier_name": null
}
```

## Technical Details

### Document Parsing
- **PDF:** `pdfplumber` — extracts tables via `page.extract_tables()` (converted to key-value text) and free text via `page.extract_text()`, with deduplication to avoid overlap between table cells and page text.
- **DOCX:** `python-docx` — extracts tables and paragraph blocks separately.
- **TXT:** Splits on double newlines into logical sections.

### Chunking Strategy
Uses LangChain's `RecursiveCharacterTextSplitter`:
- **Chunk size:** 1000 characters
- **Overlap:** 200 characters
- **Separators:** `["\n\n", "\n", ". ", " ", ""]` (splits at the most semantically meaningful boundary first)

Each parsed section (table or text block) becomes a LangChain `Document`, then gets split further if it exceeds 1000 chars. Metadata (`doc_id`, `filename`, `chunk_index`) is preserved on every chunk.

### Retrieval Method
- **Embedding model:** `gemini-embedding-001` (3072 dimensions)
- **Vector store:** ChromaDB with cosine similarity (`hnsw:space: cosine`)
- **Retrieval:** Top-3 chunks by cosine similarity
- Query is embedded → searched against the document's ChromaDB collection → top chunks returned with similarity scores

### Guardrails Approach
1. **Retrieval threshold:** If the best chunk's cosine similarity is below **0.3**, the system refuses to answer and returns `"Not found in document."` without calling the LLM — preventing hallucination on irrelevant queries.
2. **System prompt grounding:** The LLM system prompt strictly instructs: *"ONLY use information present in the provided context. If the answer is not found, respond exactly: 'Not found in document.'"*
3. **Dual context:** Both the top-3 retrieved chunks AND full document text are passed to the LLM, so it can verify answers against the complete document.

### Confidence Scoring Method
- **Score = average cosine similarity** of the top-3 retrieved chunks (0.0 to 1.0).
- Displayed as: **High** (>= 0.65), **Medium** (>= 0.45), **Low** (< 0.45) in the UI.
- Based on retrieval similarity rather than LLM self-assessment, making it an independent, objective measure of how well the query matched the document content.

### Known Failure Cases
1. **Scanned PDFs / image-only PDFs:** pdfplumber extracts text-layer only. OCR (e.g., Tesseract) would be needed for scanned documents.
2. **Very large documents:** All chunks are sent as full text to the LLM. Documents exceeding the context window (~1M tokens for Gemini) would need summarization or selective context.
3. **Multi-document queries:** Each document is stored in its own ChromaDB collection. Cross-document questions are not supported.
4. **Table-heavy documents:** Complex nested or merged-cell tables may not parse perfectly with pdfplumber's `extract_tables()`.
5. **Free-tier rate limits:** Gemini 2.5 Flash on free tier has per-minute request limits that may throttle under heavy use.

### Improvement Ideas
1. **OCR support** — Add Tesseract/EasyOCR for scanned PDF handling.
2. **Hybrid retrieval** — Combine vector similarity with BM25 keyword search for better recall on exact terms (e.g., shipment IDs, dates).
3. **Re-ranking** — Add a cross-encoder re-ranker after initial retrieval to improve precision.
4. **Streaming responses** — Stream LLM answers via SSE for better UX on longer answers.
5. **Multi-document support** — Allow querying across multiple uploaded documents.
6. **Caching** — Cache embeddings and LLM responses for repeated queries on the same document.
7. **Auth & multi-tenancy** — Add user authentication and per-user document isolation for production use.
8. **Better confidence scoring** — Use chunk agreement (do multiple chunks say the same thing?) or LLM self-evaluation as additional signals.
