"""Tests for the HTTP API, with the answer pipeline and the query log replaced by fakes."""
from contextlib import nullcontext

import pytest
from fastapi.testclient import TestClient

import api
from rag.answer import REFUSAL, Answer, Citation
from rag.retrieve import Chunk

SOURCES = [
    Chunk(id=11, document="Annual-leave", page=1, text=" Four weeks of leave. ", similarity=0.61),
    Chunk(id=12, document="Annual-leave", page=2, text="Shiftworkers get five.", similarity=0.55),
    Chunk(id=13, document="Public-holidays", page=1, text="Public holidays.", similarity=0.40),
    Chunk(id=14, document="Unpaid-work", page=3, text="Trials.", similarity=0.38),
    Chunk(id=15, document="Minimum-wages", page=1, text="Found by full text only.", similarity=None),
]


def make_answer(refused: bool) -> Answer:
    """Build a fixed Answer, cited or refused, over the same five sources."""
    citations = [] if refused else [Citation("Annual-leave", 2, "Shiftworkers get five.", 12), Citation("Annual-leave", 1, "Four weeks of leave.", 11)]
    return Answer(
        text=REFUSAL if refused else "Four weeks.",
        citations=citations,
        refused=refused,
        error=False,
        confidence=None if refused else "high",
        sources=SOURCES,
        best_similarity=0.61,
        prompt_tokens=100,
        completion_tokens=10,
        embed_tokens=5,
        cost_usd=0.0000561,
        latency_ms=42,
    )


@pytest.fixture
def client(monkeypatch):
    """A test client whose /ask never touches OpenAI or PostgreSQL; logged rows are collected on client.logged."""
    logged = []
    monkeypatch.setattr(api, "answer", lambda question: make_answer(refused=question.startswith("refuse")))
    monkeypatch.setattr(api.db, "connect", lambda: nullcontext(None))
    monkeypatch.setattr(api.db, "log_query", lambda conn, row: logged.append(row))
    test_client = TestClient(api.app)
    test_client.logged = logged
    return test_client


def test_index_serves_the_demo_page(client):
    """GET / returns the HTML page with the title."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>au-workplace-assistant</title>" in response.text


def test_ask_returns_documented_fields(client):
    """POST /ask returns answer, confidence, ranked citations, the five sources with cited flags, latency and cost."""
    data = client.post("/ask", json={"question": "How many weeks?"}).json()
    assert data["answer"] == "Four weeks." and data["refused"] is False and data["confidence"] == "high"
    assert data["citations"] == [
        {"rank": 2, "document": "Annual-leave", "page": 2, "snippet": "Shiftworkers get five."},
        {"rank": 1, "document": "Annual-leave", "page": 1, "snippet": "Four weeks of leave."},
    ]
    assert [s["rank"] for s in data["sources"]] == [1, 2, 3, 4, 5]
    assert [s["cited"] for s in data["sources"]] == [True, True, False, False, False]
    assert data["sources"][0]["snippet"] == "Four weeks of leave."
    assert data["sources"][0]["similarity"] == 0.61 and data["sources"][4]["similarity"] is None
    assert data["latency_ms"] == 42 and data["cost_usd"] == 0.000056
    assert client.logged[0]["cited_chunks"] == [12, 11] and client.logged[0]["refused"] is False


def test_refusal_has_no_citations_and_no_confidence(client):
    """A refused answer still lists the retrieved sources, none of them cited."""
    data = client.post("/ask", json={"question": "refuse this"}).json()
    assert data["refused"] is True and data["answer"] == REFUSAL and data["confidence"] is None
    assert data["citations"] == []
    assert all(s["cited"] is False for s in data["sources"]) and len(data["sources"]) == 5


def test_invalid_questions_are_422(client):
    """Empty and over-long questions are rejected before the pipeline runs."""
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422
    assert client.logged == []
