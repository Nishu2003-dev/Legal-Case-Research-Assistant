"""
ingest.py - read judgment PDFs, clean the text, split into chunks,
embed them and save everything in a local ChromaDB folder.

Run:  python ingest.py          (all PDFs in data/pdfs)
      python ingest.py 40       (only the first 40 PDFs, good for testing)
"""
import re
import sys
from pathlib import Path

import pymupdf  # PyMuPDF
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data/pdfs")
DB_DIR = "chroma_db"
COLLECTION = "legal_cases"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 200    # characters shared between neighbouring chunks
MIN_PAGE_CHARS = 200   # pages with less text than this are probably scans or blank
BATCH_SIZE = 256


def clean_text(text: str) -> str:
    """Remove the usual PDF noise so chunks read like normal paragraphs."""
    text = text.replace("\u00ad", "")                      # soft hyphens
    text = re.sub(r"-\n(?=[a-z])", "", text)               # join words split across lines
    lines = [ln.strip() for ln in text.split("\n")]
    # drop lines that are only a page number, like "12" or "Page 12"
    lines = [ln for ln in lines if not re.fullmatch(r"(page\s*)?\d{1,4}", ln, flags=re.I)]
    text = "\n".join(lines)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)           # single line breaks -> space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[Document]:
    """One Document per page, so every chunk keeps an exact page number."""
    pages = []
    with pymupdf.open(path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text())
            if len(text) < MIN_PAGE_CHARS:
                continue
            pages.append(Document(
                page_content=text,
                metadata={"source": path.name, "page": page_num},
            ))
    return pages


def main(limit: int | None = None) -> None:
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]
    if not pdfs:
        sys.exit(f"No PDFs found in {DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", ". ", " ", ""],
    )

    chunks, skipped = [], []
    for pdf in pdfs:
        pages = load_pdf(pdf)
        if not pages:
            skipped.append(pdf.name)
            continue
        pdf_chunks = splitter.split_documents(pages)
        for i, chunk in enumerate(pdf_chunks):
            chunk.metadata["chunk_id"] = f"{pdf.stem}-{i}"
        chunks.extend(pdf_chunks)

    print(f"PDFs read: {len(pdfs) - len(skipped)} | skipped (no text): {len(skipped)}")
    print(f"Chunks created: {len(chunks)}")
    if skipped:
        print("Skipped files:", ", ".join(skipped[:10]), "..." if len(skipped) > 10 else "")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        db.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
        print(f"Indexed {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"Done. Vector store saved in ./{DB_DIR}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)