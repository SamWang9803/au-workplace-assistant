"""PostgreSQL access: schema, document and chunk writes, vector and full-text search, and the query log."""
from dataclasses import dataclass

import psycopg

from rag.settings import settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id          serial PRIMARY KEY,
  name        text UNIQUE NOT NULL,      -- file name without extension
  source_url  text,
  pages       int NOT NULL,
  ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id          serial PRIMARY KEY,
  document_id int NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page        int NOT NULL,              -- 1-based page in the PDF
  idx         int NOT NULL,              -- chunk order within the page
  text        text NOT NULL,
  embedding   vector(1536) NOT NULL,
  tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);

CREATE TABLE IF NOT EXISTS queries (
  id                serial PRIMARY KEY,
  asked_at          timestamptz NOT NULL DEFAULT now(),
  question          text NOT NULL,
  refused           boolean NOT NULL,
  error             boolean NOT NULL DEFAULT false,  -- the model returned no usable JSON
  top_documents     text[] NOT NULL,
  cited_chunks      int[] NOT NULL,
  prompt_tokens     int,
  completion_tokens int,
  embed_tokens      int,
  cost_usd          numeric(10,6),
  latency_ms        int
);
"""


@dataclass
class Row:
    """A chunk returned by a search, with its score: cosine similarity or full-text rank."""

    id: int
    document: str
    page: int
    text: str
    score: float


def connect() -> psycopg.Connection:
    """Open an autocommit connection to DATABASE_URL; multi-statement writes use conn.transaction()."""
    return psycopg.connect(settings.database_url, autocommit=True, connect_timeout=5)


def init_schema(conn: psycopg.Connection) -> None:
    """Create the extension, tables and indexes if they do not exist yet."""
    conn.execute(SCHEMA)


def upsert_document(conn: psycopg.Connection, name: str, source_url: str | None, pages: int) -> int:
    """Insert or refresh a document row, delete its old chunks, and return its id."""
    doc_id = conn.execute(
        """INSERT INTO documents (name, source_url, pages) VALUES (%s, %s, %s)
           ON CONFLICT (name) DO UPDATE
             SET source_url = EXCLUDED.source_url, pages = EXCLUDED.pages, ingested_at = now()
           RETURNING id""",
        (name, source_url, pages),
    ).fetchone()[0]
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
    return doc_id


def insert_chunks(conn: psycopg.Connection, document_id: int, rows: list[tuple[int, int, str, list[float]]]) -> None:
    """Insert (page, idx, text, embedding) rows for one document."""
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (document_id, page, idx, text, embedding) VALUES (%s, %s, %s, %s, %s::vector)",
            [(document_id, page, idx, text, _vector(vector)) for page, idx, text, vector in rows],
        )


def _vector(values: list[float]) -> str:
    """Format a list of floats as a pgvector literal."""
    return "[" + ",".join(map(str, values)) + "]"


def vector_search(conn: psycopg.Connection, embedding: list[float], k: int) -> list[Row]:
    """Return the k chunks nearest to the embedding, scored by cosine similarity (1 - cosine distance)."""
    rows = conn.execute(
        """SELECT c.id, d.name, c.page, c.text, 1 - (c.embedding <=> %(v)s::vector)
           FROM chunks c JOIN documents d ON d.id = c.document_id
           ORDER BY c.embedding <=> %(v)s::vector
           LIMIT %(k)s""",
        {"v": _vector(embedding), "k": k},
    ).fetchall()
    return [Row(*row) for row in rows]


def text_search(conn: psycopg.Connection, question: str, k: int) -> list[Row]:
    """Return up to k chunks containing any of the question's stemmed words, ranked by ts_rank_cd."""
    rows = conn.execute(
        """SELECT c.id, d.name, c.page, c.text, ts_rank_cd(c.tsv, q.query) AS rank
           FROM chunks c JOIN documents d ON d.id = c.document_id,
                -- plainto_tsquery ANDs every word; OR them so a chunk needs only some of the words
                (SELECT replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | ')::tsquery AS query) q
           WHERE c.tsv @@ q.query
           ORDER BY rank DESC, c.id
           LIMIT %(k)s""",
        {"q": question, "k": k},
    ).fetchall()
    return [Row(*row) for row in rows]


def log_query(conn: psycopg.Connection, row: dict) -> None:
    """Append one answered question to the queries table."""
    conn.execute(
        """INSERT INTO queries (question, refused, error, top_documents, cited_chunks,
             prompt_tokens, completion_tokens, embed_tokens, cost_usd, latency_ms)
           VALUES (%(question)s, %(refused)s, %(error)s, %(top_documents)s, %(cited_chunks)s,
             %(prompt_tokens)s, %(completion_tokens)s, %(embed_tokens)s, %(cost_usd)s, %(latency_ms)s)""",
        row,
    )
