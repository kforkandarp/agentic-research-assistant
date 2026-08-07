import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.retriever import load_documents, build_hybrid_retriever, retrieve

_hybrid_retriever = None  # built once per program run, reused across calls


def retrieval_tool(query: str) -> str:
    """Retrieves relevant ArXiv paper excerpts for a query.
    Returns a formatted string (paper/section/page + text) ready to feed an LLM."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        documents = load_documents() # chunks.json -> Document objects
        _hybrid_retriever = build_hybrid_retriever(documents) # Produces BM25 + FAISS  -> EnsembleRetriever and Stores it globally

    results = retrieve(query, _hybrid_retriever, top_n=5)
    
    if not results:
        return "No relevant excerpts found in the local paper corpus."

    formatted = []
    for doc in results:
        meta = doc.metadata
        formatted.append(
            f"[{meta['paper']} | {meta['section']} | page {meta['page']}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


if __name__ == "__main__":
    print(retrieval_tool("What is the BLEU score of the base Transformer model?"))