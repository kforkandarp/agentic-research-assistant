"""
src/tools/retrieval_tool.py
Retrieval tool wrapper that executes hybrid retrieval + reranking thresholding
and formats output with paper metadata for LLM synthesis.
"""

import sys
import os

# Ensure package root is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.retriever import load_documents, build_hybrid_retriever, retrieve

_hybrid_retriever = None  # Global singleton, built once per session


def retrieval_tool(query: str) -> str:
    """Retrieves relevant ArXiv paper excerpts for a query.
    Returns a formatted string with metadata (paper/authors/section/page + text)."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        documents = load_documents()
        _hybrid_retriever = build_hybrid_retriever(documents)

    results = retrieve(query, _hybrid_retriever, top_n=5) # result is a list of Document objects
    
    if not results:
        return "No sufficiently relevant excerpts found in the local paper corpus."

    formatted = []
    for doc in results:
        meta = doc.metadata
        formatted.append(
            f"[{meta['paper']} | Section: {meta['section']} | Page {meta['page']} | Authors: {meta['authors']}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


if __name__ == "__main__":
    print(retrieval_tool("What is the BLEU score of the base Transformer model?"))