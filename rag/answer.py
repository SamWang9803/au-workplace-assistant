"""Answer a question from the retrieved sources with page-level citations, or refuse."""
import sys
import time
from dataclasses import dataclass

from rag.providers import chat_json, cost_usd
from rag.retrieve import Chunk, retrieve
from rag.settings import settings

REFUSAL = "The documents I have do not cover this."

SYSTEM_PROMPT = """You answer questions about Australian workplace entitlements using only the numbered sources you are given.
- Do not use outside knowledge, even if you are confident.
- Copy figures such as amounts, hours, weeks and dates exactly as the sources state them.
- List the number of every source you relied on in "citations".
- If the sources do not answer the question, say so in "answer" and return an empty "citations" list.
- Answer in one to four plain sentences.
- Set "confidence" to "high" when a source states the answer directly, "medium" when you had to combine or interpret sources, and "low" otherwise."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["answer", "citations", "confidence"],
    "additionalProperties": False,
}


@dataclass
class Citation:
    """A cited source as returned to callers."""

    document: str
    page: int
    snippet: str
    chunk_id: int


@dataclass
class Answer:
    """The answer text, citations, sources, refusal and error flags, token usage, cost and latency."""

    text: str
    citations: list[Citation]
    refused: bool
    error: bool
    confidence: str | None
    sources: list[Chunk]
    best_similarity: float
    prompt_tokens: int
    completion_tokens: int
    embed_tokens: int
    cost_usd: float
    latency_ms: int


def format_sources(chunks: list[Chunk]) -> str:
    """Number the chunks from 1 and label each with its document and page."""
    return "\n\n".join(f"[{n}] {c.document}, page {c.page}\n{c.text.strip()}" for n, c in enumerate(chunks, start=1))


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Build the user message: the numbered sources followed by the question."""
    return f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"


def validate_citations(citations: list[int], n_sources: int) -> list[int]:
    """Keep citation numbers between 1 and n_sources, without duplicates, in the model's order."""
    kept: list[int] = []
    for n in citations:
        if 1 <= n <= n_sources and n not in kept:
            kept.append(n)
    return kept


def answer(question: str) -> Answer:
    """Retrieve sources, ask the model only when retrieval found something relevant, and apply the refusal rules."""
    start = time.perf_counter()
    found = retrieve(question)
    reply = None
    if found.best_similarity >= settings.rag_min_similarity:
        reply = chat_json(SYSTEM_PROMPT, build_prompt(question, found.chunks), ANSWER_SCHEMA)
    data = reply.data if reply else None
    cited = validate_citations(data["citations"], len(found.chunks)) if data else []
    citations = [Citation(c.document, c.page, c.text.strip(), c.id) for c in (found.chunks[n - 1] for n in cited)]
    prompt_tokens = reply.prompt_tokens if reply else 0
    completion_tokens = reply.completion_tokens if reply else 0
    return Answer(
        text=data["answer"] if citations else REFUSAL,
        citations=citations,
        refused=not citations,
        error=reply is not None and data is None,
        confidence=data["confidence"] if citations else None,
        sources=found.chunks,
        best_similarity=found.best_similarity,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        embed_tokens=found.embed_tokens,
        cost_usd=cost_usd(prompt_tokens, completion_tokens, found.embed_tokens),
        latency_ms=round((time.perf_counter() - start) * 1000),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python -m rag.answer "your question"')
    result = answer(" ".join(sys.argv[1:]))
    print(result.text)
    for n, citation in enumerate(result.citations, start=1):
        print(f"  [{n}] {citation.document}, page {citation.page}")
    print(
        f"refused={result.refused} confidence={result.confidence} best_similarity={result.best_similarity:.3f} "
        f"tokens={result.prompt_tokens}+{result.completion_tokens}+{result.embed_tokens} "
        f"cost=${result.cost_usd:.6f} latency={result.latency_ms} ms"
    )
