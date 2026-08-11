"""
src/ingest.py
Bulk Ingestion & Metadata-Rich Chunking Pipeline for ArXiv Papers.

Downloads top ML papers from ArXiv via API, extracts structured metadata,
splits content with section-aware logic, and produces an enriched dataset (5000+ chunks).
"""

import os
import re
import json
import time
import logging
import urllib.request
from typing import List, Dict, Any, Optional
import arxiv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Logging Configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IngestEngine")

# ── Directories & Paths ────────────────────────────────────────────────────────
RAW_PDF_DIR = "data/raw_pdfs"
OUTPUT_CHUNKS_PATH = "data/chunks.json"

os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── ArXiv Categories & Default Search Queries ─────────────────────────
DEFAULT_SEARCH_CATEGORIES = [
    "cat:cs.CL",  # Computation and Language
    "cat:cs.LG",  # Machine Learning
    "cat:cs.AI",  # Artificial Intelligence
    "cat:cs.CV",  # Computer Vision
]

# Section Headers Regex for Section-Aware Chunking
SECTION_PATTERNS = [
    r"abstract",
    r"introduction",
    r"related work",
    r"background",
    r"method(?:ology|s)?",
    r"approach",
    r"model(?:\s+architecture)?",
    r"experiments?",
    r"results?",
    r"evaluation",
    r"discussion",
    r"conclusion",
    r"references",
]

HEADER_REGEX = re.compile(
    r"^\s*(?:\d+\.?\s*)?(" + "|".join(SECTION_PATTERNS) + r")\s*$",
    re.IGNORECASE,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)


def fetch_arxiv_papers(
    categories: List[str] = DEFAULT_SEARCH_CATEGORIES,
    max_results_per_category: int = 25
) -> List[Dict[str, Any]]:
    """
    Searches and downloads top papers from ArXiv API with metadata extraction.
    """
    logger.info(f"Initiating ArXiv bulk fetch (Target: {max_results_per_category} papers/cat)...")
    downloaded_papers = []

    client = arxiv.Client(page_size=20, delay_seconds=3, num_retries=5)

    for cat in categories:
        logger.info(f"Fetching category: {cat}")
        search = arxiv.Search(
            query=cat,
            max_results=max_results_per_category,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        try:
            for result in client.results(search):
                clean_title = re.sub(r'[\\/*?:"<>|]', "", result.title).replace(" ", "_")
                filename = f"{result.entry_id.split('/')[-1]}_{clean_title[:40]}.pdf"
                pdf_path = os.path.join(RAW_PDF_DIR, filename)

                if not os.path.exists(pdf_path):
                    logger.info(f"Downloading: {result.title[:50]}...")
                    urllib.request.urlretrieve(result.pdf_url, pdf_path)
                    time.sleep(1.0)  # Gentle API throttling

                downloaded_papers.append({
                    "paper_id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "authors": [a.name for a in result.authors[:3]],
                    "published": result.published.strftime("%Y-%m-%d"),
                    "summary": result.summary.replace("\n", " "),
                    "category": result.primary_category,
                    "pdf_path": pdf_path,
                })
        except Exception as e:
            logger.error(f"Error fetching category {cat}: {e}")

    logger.info(f"Total metadata records assembled: {len(downloaded_papers)}")
    return downloaded_papers


def split_pages_into_sections(pages) -> List[Dict[str, Any]]:
    """
    Splits page-level documents into structural paper sections.
    """
    sections = []
    current_section = "preamble"
    current_lines = []
    current_start_page = pages[0].metadata.get("page", 0) + 1 if pages else 1

    for doc in pages:
        page_num = doc.metadata.get("page", 0) + 1
        for line in doc.page_content.split("\n"):
            match = HEADER_REGEX.match(line.strip())
            if match:
                if current_lines:
                    sections.append({
                        "section": current_section,
                        "page": current_start_page,
                        "text": "\n".join(current_lines).strip(),
                    })
                current_section = match.group(1).lower()
                current_start_page = page_num
                current_lines = []
            else:
                current_lines.append(line)

    if current_lines:
        sections.append({
            "section": current_section,
            "page": current_start_page,
            "text": "\n".join(current_lines).strip(),
        })

    return [s for s in sections if len(s["text"]) > 50]


def process_and_chunk_corpus(paper_metadata_list: Optional[List[Dict[str, Any]]] = None):
    """
    Processes all PDFs in RAW_PDF_DIR, applies section splitting,
    enriches chunks with structural metadata, and outputs data/chunks.json.
    """
    all_chunks = []
    chunk_id = 0
    meta_lookup = {p["pdf_path"]: p for p in (paper_metadata_list or [])}

    pdf_files = [f for f in os.listdir(RAW_PDF_DIR) if f.endswith(".pdf")]
    logger.info(f"Beginning chunking execution over {len(pdf_files)} PDF files...")

    for filename in pdf_files:
        pdf_path = os.path.join(RAW_PDF_DIR, filename)
        paper_name = filename.replace(".pdf", "")
        meta = meta_lookup.get(pdf_path, {
            "paper_id": paper_name,
            "title": paper_name,
            "authors": ["Unknown"],
            "published": "N/A",
            "category": "cs.ML"
        })

        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            sections = split_pages_into_sections(pages)

            for section in sections:
                sub_chunks = splitter.split_text(section["text"])
                for text_chunk in sub_chunks:
                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "paper_id": meta["paper_id"],
                        "paper": meta["title"],
                        "authors": ", ".join(meta["authors"]),
                        "published": meta["published"],
                        "section": section["section"],
                        "page": section["page"],
                        "text": text_chunk,
                    })
                    chunk_id += 1
        except Exception as e:
            logger.error(f"Failed processing PDF {filename}: {e}")

    with open(OUTPUT_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    logger.info(f"Ingestion complete! Generated {len(all_chunks)} chunks.")
    logger.info(f"Artifact successfully saved to {OUTPUT_CHUNKS_PATH}")


if __name__ == "__main__":
    # Fetch 10 papers per category (40 new papers + 5 original papers = 45 total)
    metadata = fetch_arxiv_papers(max_results_per_category=10)
    process_and_chunk_corpus(metadata)