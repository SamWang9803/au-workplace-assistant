# Stage 1 implementation plan: Fair Work RAG assistant

Design: [docs/design/2026-11-02-fairwork-rag-design.md](../docs/design/2026-11-02-fairwork-rag-design.md)
Owner: Sam · Started: 2026-11-02 · Target: usable demo by 2026-11-03 evening
Status: `plan` (move to `build` when phase 0 starts, `done` when phase 8 passes)

Each phase ends with a check that proves it works. Phases are in dependency order;
0–2 need no API keys, 3 onwards need `.env`. Estimate is attention hours with
Claude Code doing the typing.

## Phase 0 · Scaffold (0.5 h)

Files: `requirements.txt`, `pyproject.toml` (pytest config only), `.env.example`,
`.gitignore` (`.env`, `data/raw/`, `evals/results/*.md` kept, `__pycache__`, `.venv`),
`docker-compose.yml` (`pgvector/pgvector:pg16`, port 5433, user/db `rag`), `README.md` skeleton.

Check:
- [ ] `python -m venv .venv` and `pip install -r requirements.txt` finish clean
- [ ] `docker compose up -d` and `docker compose exec db psql -U rag -c "select 1"` answers

## Phase 1 · Settings and providers (0.5 h)

Files: `rag/__init__.py`, `rag/settings.py`, `rag/providers.py`.

- `settings`: reads `.env`; defaults for every key listed in design §13; price table.
- `providers.embed(texts)`: OpenAI embeddings in batches, returns vectors and token usage.
- `providers.chat_json(system, user)`: Gemini via the OpenAI-compatible base URL,
  `response_format` json_object, temperature 0, minimum interval between calls,
  three retries with backoff on 429/5xx; returns parsed dict, usage, latency.

Check:
- [ ] `python -c "from rag.providers import embed; print(len(embed(['hi'])[0][0]))"` prints 1536
- [ ] `python -c "from rag.providers import chat_json; print(chat_json('Reply as JSON {\"ok\": true}', 'go'))"` returns a dict with usage and latency

## Phase 2 · Chunking with tests (0.5 h)

Files: `rag/chunking.py`, `tests/test_chunking.py`.

- `chunk_page(text, size=1800, overlap=200)`: prefer paragraph breaks, then sentence
  ends, then hard cut; consecutive chunks share `overlap` characters; no text lost.

Check:
- [ ] `pytest tests/test_chunking.py` green: size limit, overlap present, paragraph
  preference, concatenation (minus overlaps) equals the input

## Phase 3 · Database, corpus and ingest (1 h)

Files: `rag/db.py`, `scripts/fetch_corpus.py`, `rag/ingest.py`.

- `db.init_schema()` creates the extension, tables and indexes from design §5.
- `fetch_corpus.py` downloads the eleven PDFs into `data/raw/`, skips existing files,
  prints a per-file status line and exits non-zero if any download failed.
- `ingest.py`: for each PDF, pypdf page text → `chunk_page` → embed in batches of 64 →
  delete the document's old rows → insert; prints `name: pages, chunks`.

Check:
- [ ] `python scripts/fetch_corpus.py` reports 11/11 downloaded
- [ ] `python -m rag.ingest` prints eleven lines; `select count(*) from chunks` is in the low hundreds
- [ ] running ingest twice does not duplicate rows

## Phase 4 · Retrieval with tests (0.5 h)

Files: `rag/retrieve.py`, `tests/test_retrieve.py`.

- `db.vector_search(vec, k=20)` and `db.text_search(question, k=20)`.
- `rrf_merge(lists, k=60)` pure; `retrieve(question)` returns top-5 chunks and `best_similarity`.

Check:
- [ ] `pytest tests/test_retrieve.py` green: both-list items outrank single-list items, tie order stable
- [ ] `python -m rag.retrieve "How many hours is a full-time week?"` shows five chunks with
  document, page and similarity, and the maximum-weekly-hours sheet is among them

## Phase 5 · Answer with refusal and tests (0.75 h)

Files: `rag/answer.py`, `tests/test_answer.py`.

- Prompt builder with numbered sources; JSON contract from design §7.
- `validate_citations(citations, n_sources)` pure; refusal on low similarity, no valid
  citations, or unparseable JSON after one retry.
- Cost computed from usage and the price table.

Check:
- [ ] `pytest tests/test_answer.py` green
- [ ] `python -m rag.answer "How many weeks of annual leave does a full-time employee get?"`
  answers "4 weeks" with a citation to the annual leave sheet
- [ ] `python -m rag.answer "How do I apply for a 482 visa?"` refuses

## Phase 6 · API (0.5 h)

Files: `api.py`.

- `POST /ask`, `GET /health`, logging to `queries`, error mapping from design §8.

Check:
- [ ] `uvicorn api:app` starts; `curl localhost:8000/health` returns ok
- [ ] `curl -X POST localhost:8000/ask -d '{"question": "..."}'` returns the documented JSON
- [ ] the row appears in `queries`

## Phase 7 · Golden set and evaluation (1 h)

Files: `evals/golden.yaml`, `rag/eval.py`, `evals/results/`.

- Fifteen answerable questions spread across the eleven documents, three off-topic.
- Metrics from design §9; judge prompt in `eval.py`; results table written to
  `evals/results/<UTC timestamp>.md`; summary printed.
- Tune `RAG_MIN_SIMILARITY` once if a refusal or a false answer shows up; record the
  chosen value in the results file.

Check:
- [ ] `python -m rag.eval` completes within the free-tier rate limit (about 3 minutes)
- [ ] hit@5 ≥ 80%, all three off-topic questions refused, mean faithfulness ≥ 4
- [ ] the results table exists and is readable

## Phase 8 · README and final pass (0.5 h)

Files: `README.md`.

- What it is (two sentences), architecture diagram, run steps in order
  (compose, venv, env, fetch, ingest, ask, api, eval), the latest results table,
  design decisions with trade-offs (chunking, hybrid search, refusal threshold,
  citations, evaluation), what production would need next, source attribution.

Check:
- [ ] A fresh clone can follow the README top to bottom without reading the code
- [ ] `pytest` green, `python -m rag.eval` reproduces the quoted table

## Commit points (Sam commits; nothing is committed on his behalf)

1. after phase 0: `chore: scaffold project`
2. after phase 2: `feat: chunking with tests`
3. after phase 3: `feat: corpus fetch and ingest into pgvector`
4. after phase 5: `feat: hybrid retrieval and cited answers with refusal`
5. after phase 7: `feat: FastAPI /ask and golden-set evaluation`
6. after phase 8: `docs: README with results`

Total estimate: about 5.75 hours of attention, split across two sittings.
