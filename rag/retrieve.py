"""Hybrid retrieval: vector search and full-text search fused with reciprocal rank fusion."""
import sys
from dataclasses import dataclass

from rag import db
from rag.providers import embed

CANDIDATES = 20  # chunks taken from each search
TOP_N = 5  # chunks kept after fusion
RRF_K = 60


@dataclass
class Chunk:
    """A retrieved chunk: its document, page, text and vector similarity (None if only full-text search found it)."""

    id: int
    document: str
    page: int
    text: str
    similarity: float | None


@dataclass
class Retrieval:
    """The fused top chunks, the best vector similarity among the candidates, and the embedding tokens spent."""

    chunks: list[Chunk]
    best_similarity: float
    embed_tokens: int


def rrf_merge(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Fuse ranked id lists by summing 1 / (k + rank) per list; ties keep the order in which ids were first seen."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores, key=lambda item: -scores[item])


def retrieve(question: str) -> Retrieval:
    """Embed the question, run both searches and return the fused top chunks."""
    vectors, tokens = embed([question])
    with db.connect() as conn:
        by_vector = db.vector_search(conn, vectors[0], CANDIDATES)
        by_text = db.text_search(conn, question, CANDIDATES)
    similarity = {row.id: row.score for row in by_vector}
    rows = {row.id: row for row in by_vector + by_text}
    fused = rrf_merge([[row.id for row in by_vector], [row.id for row in by_text]])[:TOP_N]
    chunks = [Chunk(i, rows[i].document, rows[i].page, rows[i].text, similarity.get(i)) for i in fused]
    return Retrieval(chunks, max(similarity.values(), default=0.0), tokens)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python -m rag.retrieve "your question"')
    result = retrieve(" ".join(sys.argv[1:]))
    print(f"best similarity {result.best_similarity:.3f}")
    for n, chunk in enumerate(result.chunks, start=1):
        score = f"{chunk.similarity:.3f}" if chunk.similarity is not None else "text "
        preview = " ".join(chunk.text.split())[:90]
        print(f"[{n}] {score}  {chunk.document} p.{chunk.page}  {preview}")
