import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, GEMINI_API_KEY

_embeddings: GoogleGenerativeAIEmbeddings | None = None

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=f"models/{EMBEDDING_MODEL}",
            google_api_key=GEMINI_API_KEY,
        )
    return _embeddings


def _get_vectorstore(doc_id: str) -> Chroma:
    return Chroma(
        collection_name=f"doc_{doc_id}",
        embedding_function=_get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def store_chunks(sections: list[str], filename: str) -> tuple[str, int]:
    """Chunk sections with RecursiveCharacterTextSplitter, embed, store in ChromaDB.

    Returns (doc_id, num_chunks).
    """
    doc_id = uuid.uuid4().hex[:12]

    docs = [Document(page_content=s, metadata={"source": filename}) for s in sections]
    split_docs = _splitter.split_documents(docs)

    for i, doc in enumerate(split_docs):
        doc.metadata.update({
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
        })

    Chroma.from_documents(
        documents=split_docs,
        embedding=_get_embeddings(),
        collection_name=f"doc_{doc_id}",
        persist_directory=CHROMA_PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )

    return doc_id, len(split_docs)


def retrieve(doc_id: str, query: str, top_k: int = 3) -> list[dict]:
    """Retrieve top-k chunks by similarity.

    Returns list of {text, score, chunk_index}. Score = cosine similarity.
    """
    vectorstore = _get_vectorstore(doc_id)
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    chunks = []
    for doc, distance in results:
        similarity = 1.0 - distance
        chunks.append({
            "text": doc.page_content,
            "score": round(similarity, 4),
            "chunk_index": doc.metadata.get("chunk_index", 0),
        })

    return chunks


def get_full_text(doc_id: str) -> str:
    """Retrieve all chunks for a document, ordered, joined as full text."""
    vectorstore = _get_vectorstore(doc_id)
    all_data = vectorstore.get(include=["documents", "metadatas"])

    paired = list(zip(all_data["documents"], all_data["metadatas"]))
    paired.sort(key=lambda x: x[1]["chunk_index"])

    return "\n\n".join(doc for doc, _ in paired)
