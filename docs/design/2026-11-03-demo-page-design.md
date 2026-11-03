# Design: demo page for the Fair Work RAG assistant

Date: 2026-11-03 · Status: implemented 2026-11-03
Extends: [2026-11-02-fairwork-rag-design.md](2026-11-02-fairwork-rag-design.md) (stage 1)

## 1. Goal

A single web page for interview demos and manual testing: type a question, see
the answer with its citations, and see the five retrieved chunks with their
similarities, so hybrid retrieval and the refusal rules can be shown rather than
described. Stage 1 listed "no web UI" as a non-goal; that line becomes "a
minimal demo page; production would need a real front end".

## 2. Non-goals

Conversation history, streaming, authentication, multi-tenancy, mobile layout
polish, build tooling, external assets (fonts, CDNs).

## 3. Approach

One static file, `static/index.html`, with inline CSS and JavaScript, served by
FastAPI at `GET /`. The page calls `POST /ask` with `fetch` and renders the
JSON. No new dependencies.

Rejected: Streamlit (a second process and dependency, and it bypasses the API
the demo is meant to show); Gradio (a chat layout suggests conversation memory
the system does not have).

## 4. API changes

`POST /ask` gains backward-compatible fields:

- `rank` on each citation: the source number used in the prompt, so the page
  can label citation cards the same way the model saw them.
- `confidence`: `"high" | "medium" | "low" | null` (null when refused).
- `sources`: the fused top-5 in prompt order, each
  `{rank, document, page, snippet, similarity, cited}`. `similarity` is null
  for a chunk found only by full-text search; `cited` is true when the chunk's
  number is among the kept citations.

`GET /` returns `static/index.html`. `GET /health` is unchanged.

## 5. Page

Sections, top to bottom:

1. Header: title, a one-line description, and a small "API docs" link that
   opens `/docs` in a new tab.
2. Form: text input (max 1000 characters), an Ask button, and five example
   chips (notice period, Christmas Day, superannuation percentage, redundancy
   after 5 years, 482 visa). Clicking a chip fills the input.
3. Status strip: "Answered" in green or "Refused" in amber, then confidence,
   latency in seconds and cost in USD.
4. Answer text (the fixed refusal sentence when refused).
5. Cited sources: one card per citation, numbered as in the prompt, with
   document, page and the snippet truncated at 300 characters behind an
   expand toggle.
6. Retrieved sources: a collapsed `<details>` block with one row per source:
   rank, document, page, similarity (or "text only") and a "cited" mark.

States: idle (header and form only), loading (button disabled, label
"Asking…"), result, error. Errors show one sentence each: 422 invalid
question, 502 model provider failed, 503 database unavailable, network error.

Interface text is English. Document names appear as in the corpus (file name
without extension). Everything is rendered with `textContent`, never
`innerHTML`, because snippets are untrusted PDF text.

## 6. Tests

`tests/test_api.py` uses FastAPI's `TestClient` with `api.answer` replaced by a
fake returning a fixed `Answer` and `db.log_query` replaced by a no-op:
`GET /` returns HTML containing the title; `POST /ask` returns the documented
fields, including `sources` whose `cited` flags match the citations; an empty
question returns 422. No network, no database.

## 7. Documentation

Both READMEs: the running steps add "open http://127.0.0.1:8000/", the API
section lists the new fields, the non-goals line changes as in §1, and the
repository layout gains `static/`. The stage-1 design's non-goals get a
pointer to this document.

## 8. Acceptance

The page answers the five example questions in a browser; the visa question
shows the amber refusal; the retrieved list's "cited" marks match the citation
cards; `pytest` is green.
