"""
Hybrid retriever: BM25 (keyword) + FAISS (semantic) fused via LangChain's
EnsembleRetriever, then reranked with a cross-encoder — same architecture
as FinSight, pointed at the section-aware ArXiv chunks instead.
FAISS index is cached to disk so it's only embedded once, not on every run.
"""

import json
import os
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

CHUNKS_PATH = "data/chunks.json"
FAISS_INDEX_DIR = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # same as FinSight
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"      # same as FinSight


def load_documents() -> list[Document]:  # "I convert the JSON chunks into LangChain Document objects 
    # because both the BM25 retriever and FAISS vector store are designed to operate on the standardized Document abstraction."
    """Turns chunks.json rows into LangChain Document objects — the
    common format BM25Retriever and FAISS both expect."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    documents = []
    for chunk in raw_chunks:
        documents.append(Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "paper": chunk["paper"],
                "section": chunk["section"],
                "page": chunk["page"],
            },
        ))
    return documents




import hashlib

CHUNKS_HASH_PATH = os.path.join(FAISS_INDEX_DIR, "chunks_hash.txt")


def _compute_chunks_hash() -> str:
    """SHA-256 of chunks.json content. Changes if any paper is added,
    removed, or re-chunked — guarantees FAISS is always in sync with BM25."""
    with open(CHUNKS_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _faiss_is_valid() -> bool:
    """Returns True only if the cached FAISS index exists AND was built
    from the same chunks.json that's on disk right now."""
    if not os.path.exists(FAISS_INDEX_DIR):
        return False
    if not os.path.exists(CHUNKS_HASH_PATH):
        return False  # index exists but predates hash tracking — treat as stale
    with open(CHUNKS_HASH_PATH, "r") as f:
        cached_hash = f.read().strip()
    return cached_hash == _compute_chunks_hash()


def build_hybrid_retriever(documents: list[Document], k: int = 10) -> EnsembleRetriever:
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if _faiss_is_valid():
        print("[faiss] loading cached index from disk")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("[faiss] chunks changed or no valid cache — rebuilding index")
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(FAISS_INDEX_DIR)
        with open(CHUNKS_HASH_PATH, "w") as f:
            f.write(_compute_chunks_hash())
        print(f"[faiss] index cached to {FAISS_INDEX_DIR}")

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
    )


def rerank(query: str, candidates: list[Document], top_n: int = 5) -> list[Document]:
    """Cross-encoder reads (query, chunk) TOGETHER, giving a more accurate
    relevance score than the bi-encoder FAISS used. Slower, so we only run
    it on the small candidate set the hybrid retriever already narrowed down."""
    reranker = CrossEncoder(RERANKER_MODEL)

    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(candidates, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_n]]


def retrieve(query: str, hybrid_retriever: EnsembleRetriever, top_n: int = 5) -> list[Document]:
    """Full pipeline: hybrid retrieve -> cross-encoder rerank -> top_n."""
    candidates = hybrid_retriever.invoke(query)
    return rerank(query, candidates, top_n=top_n)


if __name__ == "__main__":
    print("[loading chunks]")
    documents = load_documents()
    print(f"  -> {len(documents)} documents loaded")

    print("[building hybrid retriever]")
    hybrid_retriever = build_hybrid_retriever(documents)

    test_query = "What method does the paper use to handle word order without recurrence?"
    print(f"\n[test query] {test_query}")

    results = retrieve(test_query, hybrid_retriever, top_n=5)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Paper: {doc.metadata['paper']} | Section: {doc.metadata['section']} | Page: {doc.metadata['page']}")