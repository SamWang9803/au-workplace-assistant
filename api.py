"""HTTP API: GET / serves the demo page, POST /ask answers a question with citations and logs it, GET /health checks the database."""
from pathlib import Path

import openai
import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag import db
from rag.answer import answer

app = FastAPI(title="au-workplace-assistant")
INDEX_PAGE = Path(__file__).resolve().parent / "static" / "index.html"


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(min_length=1, max_length=1000)


@app.post("/ask")
def ask(body: AskRequest) -> dict:
    """Answer the question, log it to the queries table, and return the answer with its citations."""
    try:
        result = answer(body.question)
        with db.connect() as conn:
            db.log_query(
                conn,
                {
                    "question": body.question,
                    "refused": result.refused,
                    "error": result.error,
                    "top_documents": list(dict.fromkeys(chunk.document for chunk in result.sources)),
                    "cited_chunks": [citation.chunk_id for citation in result.citations],
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "embed_tokens": result.embed_tokens,
                    "cost_usd": result.cost_usd,
                    "latency_ms": result.latency_ms,
                },
            )
    except psycopg.OperationalError:
        raise HTTPException(status_code=503, detail="Database unavailable.") from None
    except openai.OpenAIError:
        raise HTTPException(status_code=502, detail="The model provider failed after retries.") from None
    ranks = {chunk.id: rank for rank, chunk in enumerate(result.sources, start=1)}  # numbering used in the prompt
    cited_ids = {c.chunk_id for c in result.citations}
    return {
        "answer": result.text,
        "refused": result.refused,
        "confidence": result.confidence,
        "citations": [
            {"rank": ranks[c.chunk_id], "document": c.document, "page": c.page, "snippet": c.snippet}
            for c in result.citations
        ],
        "sources": [
            {
                "rank": ranks[chunk.id],
                "document": chunk.document,
                "page": chunk.page,
                "snippet": chunk.text.strip(),
                "similarity": None if chunk.similarity is None else round(chunk.similarity, 3),
                "cited": chunk.id in cited_ids,
            }
            for chunk in result.sources
        ],
        "latency_ms": result.latency_ms,
        "cost_usd": round(result.cost_usd, 6),
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page demo UI."""
    return FileResponse(INDEX_PAGE, media_type="text/html")


@app.get("/health")
def health() -> dict:
    """Report whether the database is reachable."""
    try:
        with db.connect() as conn:
            conn.execute("SELECT 1")
    except psycopg.OperationalError:
        raise HTTPException(status_code=503, detail="Database unavailable.") from None
    return {"status": "ok"}
