"""
src/retriever.py
Hybrid retriever: BM25 (keyword) + FAISS (semantic) fused via LangChain's
EnsembleRetriever, then reranked with a cross-encoder with SCORE THRESHOLDING.

FAISS index is cached to disk and automatically invalidated using a SHA-256
hash of data/chunks.json.
"""

import json
import os
import hashlib
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

logger = logging.getLogger("RetrieverEngine")

CHUNKS_PATH = "data/chunks.json"
FAISS_INDEX_DIR = "data/faiss_index"
CHUNKS_HASH_PATH = os.path.join(FAISS_INDEX_DIR, "chunks_hash.txt")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Minimum Cross-Encoder raw logit score required to consider a chunk relevant.
# ms-marco-MiniLM-L-6-v2 outputs raw logits (typically between -10 and +10).
# Scores below -2.5 generally indicate irrelevant noise.
RERANK_SCORE_THRESHOLD = -2.5


def load_documents() -> List[Document]:
    """Turns chunks.json rows into LangChain Document objects with rich metadata."""
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Missing {CHUNKS_PATH}. Run 'python -m src.ingest' first.")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    documents = []
    for chunk in raw_chunks:
        documents.append(Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk.get("chunk_id"),
                "paper": chunk.get("paper", "Unknown"),
                "authors": chunk.get("authors", "Unknown"),
                "published": chunk.get("published", "N/A"),
                "section": chunk.get("section", "preamble"),
                "page": chunk.get("page", 1),
            },
        ))
    return documents


def _compute_chunks_hash() -> str:
    """SHA-256 of chunks.json content. Guarantees FAISS index sync."""
    with open(CHUNKS_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _faiss_is_valid() -> bool:
    """Checks if cached FAISS index matches current data/chunks.json."""
    if not os.path.exists(FAISS_INDEX_DIR) or not os.path.exists(CHUNKS_HASH_PATH):
        return False
    with open(CHUNKS_HASH_PATH, "r") as f:
        cached_hash = f.read().strip()
    return cached_hash == _compute_chunks_hash()


def build_hybrid_retriever(documents: List[Document], k: int = 15) -> EnsembleRetriever:
    """Builds 50/50 EnsembleRetriever combining BM25 and FAISS dense search."""
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if _faiss_is_valid():
        logger.info("[FAISS] Loading cached vector index from disk...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    else:
        logger.info("[FAISS] Invalidation detected / missing cache — rebuilding FAISS vector index...")
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(FAISS_INDEX_DIR)
        with open(CHUNKS_HASH_PATH, "w") as f:
            f.write(_compute_chunks_hash())
        logger.info(f"[FAISS] Vector index rebuilt & cached at {FAISS_INDEX_DIR}")

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
    )


def rerank(
    query: str,
    candidates: List[Document],
    top_n: int = 5,
    score_threshold: float = RERANK_SCORE_THRESHOLD
) -> List[Document]:
    """
    Reranks candidates using Cross-Encoder AND applies score thresholding to filter noise.
    """
    if not candidates:
        return []

    reranker = CrossEncoder(RERANKER_MODEL)
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(candidates, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Filter out chunks below the relevance threshold
    filtered_docs = [doc for doc, score in scored_docs if score >= score_threshold]

    # If no docs meet threshold, return empty rather than bad context
    return filtered_docs[:top_n]


def retrieve(
    query: str,
    hybrid_retriever: EnsembleRetriever,
    top_n: int = 5
) -> List[Document]:
    """Full Pipeline: Hybrid Retrieve -> Cross-Encoder Rerank + Thresholding -> Top N."""
    candidates = hybrid_retriever.invoke(query)
    return rerank(query, candidates, top_n=top_n)


if __name__ == "__main__":
    docs = load_documents()
    retriever_inst = build_hybrid_retriever(docs)
    test_q = "What method does Attention Is All You Need use for word order?"
    res = retrieve(test_q, retriever_inst, top_n=3)
    print(f"\nQuery: {test_q}\nRetrieved {len(res)} chunks after thresholding:")
    for d in res:
        print(f" - [{d.metadata['paper']} | {d.metadata['section']}]: {d.page_content[:100]}...")