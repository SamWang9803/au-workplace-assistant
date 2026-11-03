# Stage 1 implementation plan: Fair Work RAG assistant

Design: [docs/design/2026-11-02-fairwork-rag-design.md](../design/2026-11-02-fairwork-rag-design.md)
Owner: Sam · Started: 2026-11-02 · Target: usable demo by 2026-11-03 evening
Status: `done` 2026-11-02, apart from the fresh-clone walk-through noted in phase 8.

Each phase ends with a check that proves it works. Phases are in dependency order;
0–2 need no API keys, 3 onwards need `.env`. Estimate is attention hours with
Claude Code doing the typing.

## Phase 0 · Scaffold (0.5 h)

Files: `requirements.txt`, `pyproject.toml` (pytest config only),
`docker-compose.yml` (`pgvector/pgvector:pg16`, port 5433, user/db `rag`).
Already in place: `.env.example`, `.gitignore` (`.env`, `data/raw/`; add `__pycache__`
and `.venv`, keep `evals/results/*.md`), `README.md`, `README_CN.md`.

Check:
- [x] `python -m venv .venv` and `pip install -r requirements.txt` finish clean
- [x] `docker compose up -d` and `docker compose exec db psql -U rag -c "select 1"` answers

## Phase 1 · Settings and providers (0.5 h)

Files: `rag/__init__.py`, `rag/settings.py`, `rag/providers.py`.

- `settings`: reads `.env`; defaults for every key listed in design §13; price table.
- `providers.embed(texts)`: OpenAI embeddings in batches, returns vectors and token usage.
- `providers.chat_json(system, user, schema)`: OpenAI `CHAT_MODEL` (default `gpt-4.1-mini`),
  `response_format` strict `json_schema` built from `schema`, temperature 0,
  the SDK's three retries with backoff on 429/5xx; returns
  `ChatReply(data, prompt_tokens, completion_tokens, latency_ms)`, `data` None if unusable.

Check:
- [x] `python -c "from rag.providers import embed; print(len(embed(['hi'])[0][0]))"` prints 1536
- [x] `python -c "from rag.providers import chat_json; print(chat_json('Reply with ok set to true.', 'go', {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok'], 'additionalProperties': False}))"` prints `{'ok': True}` with usage and latency

## Phase 2 · Chunking with tests (0.5 h)

Files: `rag/chunking.py`, `tests/test_chunking.py`.

- `chunk_page(text, size=1800, overlap=200)`: prefer paragraph breaks, then sentence
  ends, then hard cut; consecutive chunks share `overlap` characters; no text lost.

Check:
- [x] `pytest tests/test_chunking.py` green: size limit, overlap present, paragraph
  preference, concatenation (minus overlaps) equals the input

## Phase 3 · Database, corpus and ingest (1 h)

Files: `rag/db.py`, `scripts/fetch_corpus.py`, `rag/ingest.py`.

- `db.init_schema()` creates the extension, tables and indexes from design §5.
- `fetch_corpus.py` downloads the eleven PDFs into `data/raw/`, skips existing files,
  prints a per-file status line and exits non-zero if any download failed.
- `ingest.py`: for each PDF, pypdf page text → `clean_text` → `chunk_page` → embed in batches of 64 →
  delete the document's old rows → insert; prints `name: pages, chunks`.

Note 2026-11-02: the automated download stalled when run from Claude's sandbox and was
not retried there. Sam runs `python scripts/fetch_corpus.py` in his own terminal, or saves
the eleven PDFs from a browser into `data/raw/` (the URLs are in `rag/settings.py`).
Sam downloaded all eleven by hand the same evening.

Check:
- [x] all eleven PDFs are in `data/raw/` (downloaded by hand, so the script itself is still untested)
- [x] `python -m rag.ingest` prints eleven lines; `select count(*) from chunks` is 73, fewer than guessed
- [x] running ingest twice does not duplicate rows

## Phase 4 · Retrieval with tests (0.5 h)

Files: `rag/retrieve.py`, `tests/test_retrieve.py`.

- `db.vector_search(vec, k=20)` and `db.text_search(question, k=20)`.
- `rrf_merge(lists, k=60)` pure; `retrieve(question)` returns top-5 chunks and `best_similarity`.

Check:
- [x] `pytest tests/test_retrieve.py` green: both-list items outrank single-list items, tie order stable
- [x] `python -m rag.retrieve "How many hours is a full-time week?"` shows five chunks with
  document, page and similarity, and the maximum-weekly-hours sheet is among them

## Phase 5 · Answer with refusal and tests (0.75 h)

Files: `rag/answer.py`, `tests/test_answer.py`.

- Prompt builder with numbered sources; JSON contract from design §7.
- `validate_citations(citations, n_sources)` pure; refusal on low similarity, no valid
  citations, or a reply without usable JSON (no retry: with a strict schema that only
  means a safety refusal or a cut-off reply).
- Cost computed from usage and the price table.

Check:
- [x] `pytest tests/test_answer.py` green
- [x] `python -m rag.answer "How many weeks of annual leave does a full-time employee get?"`
  answers "4 weeks" with a citation to the annual leave sheet
- [x] `python -m rag.answer "How do I apply for a 482 visa?"` refuses (checked in the evaluation run)

## Phase 6 · API (0.5 h)

Files: `api.py`.

- `POST /ask`, `GET /health`, logging to `queries`, error mapping from design §8.

Check:
- [x] `uvicorn api:app` starts; `curl 127.0.0.1:8000/health` returns ok
- [x] `curl -X POST 127.0.0.1:8000/ask -d '{"question": "..."}'` returns the documented JSON
  (checked on the empty index, which refuses; 422, 502 and 503 mappings checked too)
- [x] the row appears in `queries`

## Phase 7 · Golden set and evaluation (1 h)

Files: `evals/golden.yaml` (drafted from the NES, then every gold answer checked against the
PDFs on 2026-11-02), `rag/eval.py`, `evals/results/`.

- Fifteen answerable questions spread across the eleven documents, three off-topic.
- Metrics from design §9; judge prompt in `eval.py`; results table written to
  `evals/results/<UTC timestamp>.md`; summary printed.
- Run `python -m rag.eval --retrieval-only` first (embedding cost only) to check hit@5 and
  to read the similarity gap between answerable and off-topic questions.
- Tune `RAG_MIN_SIMILARITY` once if a refusal or a false answer shows up; record the
  chosen value in the results file.

Check:
- [x] `python -m rag.eval` completes in about a minute and cost USD 0.029 (31 chat calls, 18 embeddings)
- [x] hit@5 100%, all three off-topic questions refused, mean faithfulness 4.80; the one wrong
  answer traces to pypdf scrambling the redundancy table, and `RAG_MIN_SIMILARITY` stays 0.30
- [x] the results table exists and is readable

## Phase 8 · README and final pass (0.5 h)

Files: `README.md`, `README_CN.md` (both already written ahead of the code).

- Paste the latest results table and the actual spend from the OpenAI usage
  dashboard into both READMEs.
- Check every command, module name and path in the READMEs against the real code
  and fix whichever side is wrong.

Check:
- [ ] A fresh clone can follow the README top to bottom without reading the code (every
  command was run in this working copy on 2026-11-02, but not from a fresh clone)
- [x] `pytest` green; the quoted table is the 2026-11-02 run in `evals/results/`
- [x] README.md and README_CN.md say the same thing

## Commit points (Sam commits; nothing is committed on his behalf)

1. after phase 0: `chore: scaffold project`
2. after phase 2: `feat: chunking with tests`
3. after phase 3: `feat: corpus fetch and ingest into pgvector`
4. after phase 5: `feat: hybrid retrieval and cited answers with refusal`
5. after phase 7: `feat: FastAPI /ask and golden-set evaluation`
6. after phase 8: `docs: README with results`

Total estimate: about 5.75 hours of attention, split across two sittings.
