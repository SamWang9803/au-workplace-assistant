"""Tests for citation validation and the refusal rules, with retrieval and the chat model replaced by fakes."""
import pytest

from rag import answer as answer_module
from rag.answer import REFUSAL, answer, build_prompt, validate_citations
from rag.providers import ChatReply
from rag.retrieve import Chunk, Retrieval
from rag.settings import settings

CHUNKS = [Chunk(id=10 + n, document=f"Doc-{n}", page=n, text=f"Source text {n}.", similarity=0.5) for n in range(1, 6)]
HIGH = settings.rag_min_similarity + 0.2
LOW = settings.rag_min_similarity - 0.01


def fake_pipeline(monkeypatch, similarity: float, reply: dict | None) -> list[str]:
    """Replace retrieval and the chat model with fakes; return a list that records every chat call."""
    calls: list[str] = []

    def fake_chat(system: str, user: str, schema: dict) -> ChatReply:
        """Record the call and return the canned reply."""
        calls.append(user)
        return ChatReply(reply, prompt_tokens=1000, completion_tokens=50, latency_ms=5)

    monkeypatch.setattr(answer_module, "retrieve", lambda question: Retrieval(CHUNKS, similarity, embed_tokens=7))
    monkeypatch.setattr(answer_module, "chat_json", fake_chat)
    return calls


def test_validate_citations_drops_out_of_range_numbers_and_duplicates():
    """Numbers outside 1..n and repeats are dropped; the model's order is kept."""
    assert validate_citations([0, 3, 6, 1, 3, -2], 5) == [3, 1]


def test_build_prompt_numbers_sources_from_one():
    """Sources are numbered from 1 with their document and page, followed by the question."""
    prompt = build_prompt("How long?", CHUNKS[:2])
    assert "[1] Doc-1, page 1\nSource text 1." in prompt
    assert "[2] Doc-2, page 2\nSource text 2." in prompt
    assert prompt.endswith("Question: How long?")


def test_low_similarity_refuses_without_calling_the_model(monkeypatch):
    """Below the threshold the answer is refused and the chat model is never called."""
    calls = fake_pipeline(monkeypatch, LOW, {"answer": "x", "citations": [1], "confidence": "high"})
    result = answer("q")
    assert result.refused and result.text == REFUSAL and not result.error
    assert calls == []
    assert result.cost_usd == pytest.approx(7 * settings.price_embed / 1_000_000)


def test_empty_citations_refuse(monkeypatch):
    """An answer without citations is replaced by the refusal."""
    fake_pipeline(monkeypatch, HIGH, {"answer": "Not covered.", "citations": [], "confidence": "low"})
    result = answer("q")
    assert result.refused and result.text == REFUSAL and not result.error


def test_only_out_of_range_citations_refuse(monkeypatch):
    """Citations that point at no source count as no citations."""
    fake_pipeline(monkeypatch, HIGH, {"answer": "x", "citations": [0, 9], "confidence": "high"})
    assert answer("q").refused


def test_unusable_reply_refuses_and_flags_an_error(monkeypatch):
    """A reply without usable JSON is refused and marked as an error."""
    fake_pipeline(monkeypatch, HIGH, None)
    result = answer("q")
    assert result.refused and result.error


def test_valid_citations_map_back_to_their_chunks(monkeypatch):
    """Kept citation numbers become document, page and chunk id; cost covers chat and embedding tokens."""
    fake_pipeline(monkeypatch, HIGH, {"answer": "Four weeks.", "citations": [2, 7, 2], "confidence": "high"})
    result = answer("q")
    assert not result.refused and result.text == "Four weeks." and result.confidence == "high"
    assert [(c.document, c.page, c.chunk_id) for c in result.citations] == [("Doc-2", 2, 12)]
    expected = (1000 * settings.price_chat_input + 50 * settings.price_chat_output + 7 * settings.price_embed) / 1e6
    assert result.cost_usd == pytest.approx(expected)
