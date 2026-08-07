"""
Section-aware chunking for ArXiv PDFs.
Loads pages with PyPDFLoader (same as FinSight), detects ArXiv section
headers to split each paper into sections FIRST, then applies
RecursiveCharacterTextSplitter WITHIN each section — so no chunk
straddles a section boundary. Each chunk keeps a page number for citation.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re
import json

RAW_DIR = "data/raw_pdfs"
OUTPUT_PATH = "data/chunks.json"

# Common ArXiv section header patterns. Loose and case-insensitive since
# formatting varies paper to paper.
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

# Matches lines like "3. Related Work" or "3 Related Work" or "Related Work"
HEADER_REGEX = re.compile(
    r"^\s*(?:\d+\.?\s*)?(" + "|".join(SECTION_PATTERNS) + r")\s*$",
    re.IGNORECASE,
)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)


def split_pages_into_sections(pages) -> list[dict]:
    """
    pages: list of LangChain Document objects from PyPDFLoader, one per page.
    Walks line by line across pages. A header line starts a new section;
    everything until the next header belongs to the previous one.
    Tracks the page number a section STARTED on, for citation purposes.
    """
    sections = []
    current_section = "preamble"
    current_lines = []
    current_start_page = pages[0].metadata.get("page", 0) + 1 if pages else 1

    for doc in pages:
        page_num = doc.metadata.get("page", 0) + 1  # PyPDFLoader pages are 0-indexed
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

    # Drop headers with basically no content following them
    return [s for s in sections if len(s["text"]) > 50]


def process_all_papers():
    all_chunks = []
    chunk_id = 0

    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".pdf"):
            continue

        paper_name = filename.replace(".pdf", "")
        print(f"[processing] {filename}")

        loader = PyPDFLoader(os.path.join(RAW_DIR, filename))
        pages = loader.load()
        sections = split_pages_into_sections(pages)

        for section in sections:
            sub_chunks = splitter.split_text(section["text"])
            for text_chunk in sub_chunks:
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "paper": paper_name,
                    "section": section["section"],
                    "page": section["page"],
                    "text": text_chunk,
                })
                chunk_id += 1

        print(f"  -> {len(sections)} sections detected")

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    process_all_papers()