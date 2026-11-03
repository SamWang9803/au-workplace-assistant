"""Ingest PDFs: extract text page by page, chunk it, embed the chunks in batches and store them."""
import sys
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from rag import db
from rag.chunking import chunk_page, clean_text
from rag.providers import cost_usd, embed
from rag.settings import CORPUS_URLS, settings

BATCH = 64


def read_pages(path: Path) -> list[str]:
    """Return the cleaned text of every page of a PDF."""
    return [clean_text(page.extract_text() or "") for page in PdfReader(path).pages]


def ingest_file(conn, path: Path) -> tuple[int, int, int]:
    """Chunk, embed and store one PDF, replacing any earlier copy; return pages, chunks and embedding tokens."""
    pages = read_pages(path)
    rows = []
    for page_no, text in enumerate(pages, start=1):
        if not text:
            print(f"  warning: {path.name} page {page_no} has no extractable text")
        chunks = chunk_page(text, settings.chunk_size, settings.chunk_overlap)
        rows += [(page_no, idx, chunk) for idx, chunk in enumerate(chunks)]
    vectors, tokens = [], 0
    for i in range(0, len(rows), BATCH):  # embed before touching the database, so a failure keeps the old copy
        batch_vectors, batch_tokens = embed([text for _, _, text in rows[i : i + BATCH]])
        vectors += batch_vectors
        tokens += batch_tokens
    with conn.transaction():
        doc_id = db.upsert_document(conn, path.stem, CORPUS_URLS.get(path.name), len(pages))
        db.insert_chunks(conn, doc_id, [(page, idx, text, vector) for (page, idx, text), vector in zip(rows, vectors)])
    return len(pages), len(rows), tokens


def main(folder: str = "data/raw") -> None:
    """Ingest every PDF in the folder, skipping unreadable files, and print one line per document."""
    paths = sorted(Path(folder).glob("*.pdf"))
    if not paths:
        sys.exit(f"No PDFs in {folder}: run `python scripts/fetch_corpus.py` first.")
    total_tokens = 0
    with db.connect() as conn:
        db.init_schema(conn)
        for path in paths:
            try:
                pages, chunks, tokens = ingest_file(conn, path)
            except PyPdfError as exc:
                print(f"skipped {path.name}: unreadable PDF ({exc})")
                continue
            total_tokens += tokens
            print(f"{path.stem}: {pages} pages, {chunks} chunks")
    print(f"embedded {total_tokens} tokens, about ${cost_usd(0, 0, total_tokens):.4f} at list price")


if __name__ == "__main__":
    main(*sys.argv[1:2])
