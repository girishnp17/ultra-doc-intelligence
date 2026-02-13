import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import GEMINI_API_KEY, LLM_MODEL

_ask_llm: ChatGoogleGenerativeAI | None = None
_extract_llm: ChatGoogleGenerativeAI | None = None


def _get_ask_llm() -> ChatGoogleGenerativeAI:
    global _ask_llm
    if _ask_llm is None:
        _ask_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
        )
    return _ask_llm


def _get_extract_llm() -> ChatGoogleGenerativeAI:
    global _extract_llm
    if _extract_llm is None:
        _extract_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.0,
        )
    return _extract_llm


_ask_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a logistics document assistant. You answer questions strictly "
     "based on the provided document context.\n\n"
     "Rules:\n"
     "- ONLY use information present in the provided context to answer.\n"
     '- If the answer is not found in the context, respond exactly: "Not found in document."\n'
     "- Do not speculate, infer, or use external knowledge.\n"
     "- When quoting, reference the exact text from the context.\n"
     "- Be concise and precise."),
    ("human",
     "## Retrieved Context (most relevant sections)\n{context}\n\n"
     "## Full Document Text\n{full_text}\n\n"
     "## Question\n{question}"),
])

_extract_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a logistics document data extractor. Extract structured shipment "
     "information from the provided document text.\n\n"
     "Return a JSON object with these fields (use null for any field not found):\n"
     "- shipment_id: string\n"
     "- shipper: string (company name)\n"
     "- consignee: string (company name)\n"
     "- pickup_datetime: string (ISO 8601 or as stated in document)\n"
     "- delivery_datetime: string (ISO 8601 or as stated in document)\n"
     '- equipment_type: string (e.g., "Dry Van", "Reefer", "Flatbed")\n'
     '- mode: string (e.g., "FTL", "LTL")\n'
     "- rate: number (numeric value only)\n"
     '- currency: string (e.g., "USD")\n'
     "- weight: string (include unit)\n"
     "- carrier_name: string\n\n"
     "Rules:\n"
     "- ONLY extract information explicitly stated in the document.\n"
     "- Use null for any field not found — do NOT guess.\n"
     "- Return ONLY the JSON object, no extra text."),
    ("human",
     "## Document Text\n{full_text}\n\n"
     "Extract the structured shipment information as JSON."),
])


def ask_question(
    query: str, retrieved_chunks: list[dict], full_text: str
) -> dict:
    """Ask Gemini a question grounded in retrieved context.

    Returns {answer, source_text, confidence_score}.
    """
    chunk_texts = [c["text"] for c in retrieved_chunks]
    context_block = "\n\n---\n\n".join(chunk_texts)

    chain = _ask_prompt | _get_ask_llm() | StrOutputParser()
    answer = chain.invoke({
        "context": context_block,
        "full_text": full_text,
        "question": query,
    })

    scores = [c["score"] for c in retrieved_chunks]
    confidence = round(sum(scores) / len(scores), 4) if scores else 0.0
    source_text = chunk_texts[0] if chunk_texts else ""

    return {
        "answer": answer.strip(),
        "source_text": source_text,
        "confidence_score": confidence,
    }


def extract_structured(full_text: str) -> dict:
    """Extract structured shipment data from the full document text.

    Returns a JSON dict with shipment fields (nulls for missing).
    """
    chain = _extract_prompt | _get_extract_llm() | StrOutputParser()
    raw = chain.invoke({"full_text": full_text}).strip()

    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_response": raw, "error": "Failed to parse LLM JSON output"}
