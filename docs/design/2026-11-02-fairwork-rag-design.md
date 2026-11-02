# Design: Fair Work RAG assistant (stage 1)

Date: 2026-11-02 · Status: approved in conversation, pending implementation

Revision 2026-11-02: embeddings moved from OpenAI `text-embedding-3-small` to
Gemini `gemini-embedding-2`; the project now uses a single provider and key.

## 1. Goal

Answer questions about Australian minimum workplace entitlements from Fair Work
Ombudsman (FWO) fact sheets, with every answer citing the document and page it
came from, refusing when the documents do not cover the question, and with an
evaluation table that shows how well it does. The project is a personal
portfolio piece: every component must be explainable in an interview.

Stage 2 (later, out of scope here) adds an agent with tools and human-in-the-loop
on top of the same retrieval and answer modules.

### Non-goals for stage 1

No web UI, no authentication, no multi-tenancy, no reranker, no caching, no
conversation memory. These are listed in the README as "what production would
need next".

## 2. Corpus

Eleven FWO PDFs, downloaded at build time by `scripts/fetch_corpus.py` into
`data/raw/` (git-ignored, not redistributed). All verified reachable on
2026-11-02 under `https://www.fairwork.gov.au/sites/default/files/migration/`:

| File | Path |
|---|---|
| Introduction to the National Employment Standards | `723/Introduction-to-the-national-employment-standards.pdf` |
| Maximum weekly hours | `723/Maximum-weekly-hours.pdf` |
| Annual leave | `723/Annual-leave.pdf` |
| Public holidays | `723/Public-holidays.pdf` |
| Notice of termination and redundancy pay | `723/Notice-of-termination-and-redundancy-pay.pdf` |
| Minimum wages | `723/Minimum-wages.pdf` |
| Long service leave | `723/Long-service-leave.pdf` |
| Community service leave | `723/Community-service-leave.pdf` |
| Unpaid work | `723/Unpaid-work.pdf` |
| Record-keeping and pay slips | `723/Record-keeping-and-pay-slips.pdf` |
| Fair Work Information Statement | `724/Fair-Work-Information-Statement.pdf` |

Any additional `.pdf` dropped into `data/raw/` is ingested the same way. Source
material is © Commonwealth of Australia (Fair Work Ombudsman); the repository
stores only derived chunks locally and credits the source in the README.

## 3. Architecture

Single Python 3.12 process plus one PostgreSQL 16 container with pgvector.
No orchestration framework: the pipeline is hand-written so each step is
visible and testable.

```
PDFs ──ingest──▶ chunks (text, embedding, tsvector) in PostgreSQL
question ──embed──▶ vector top-20 ─┐
question ──websearch_to_tsquery──▶ full-text top-20 ─┴─ RRF ──▶ top-5 chunks
top-5 + question ──prompt──▶ Gemini (JSON) ──validate citations / refuse──▶ answer
every /ask ──▶ queries table (tokens, cost, latency)
```

All model access goes through the `openai` SDK pointed at Gemini's
OpenAI-compatible endpoint `https://generativelanguage.googleapis.com/v1beta/openai/`
(env `GEMINI_BASE_URL`), with the single key `GEMINI_API_KEY`:

| Use | Model (default, env-overridable) |
|---|---|
| Chat and judge | `gemini-2.5-flash`, temperature 0, `response_format` json_object |
| Embeddings | `gemini-embedding-2` with `dimensions=1536`: the model is Matryoshka-trained, so the server truncates its 3072-dim output to 1536 and returns it normalised. 1536 keeps the `vector(1536)` column and its HNSW index (pgvector indexes `vector` up to 2000 dims). Verified against the live endpoint on 2026-11-02 |

Both models are free of charge on the Gemini free tier. The chat limit is
roughly 5 to 10 requests per minute, so the chat client enforces a minimum
interval between calls (default 7 s, env `CHAT_MIN_INTERVAL_S`); both chat and
embedding calls retry 429/5xx three times with exponential backoff. Embedding
limits are per project and shown in AI Studio; batches of 32 chunks keep each
request well under the reported 30k tokens-per-minute free limit. The free
tier may use inputs to improve Google's products; the corpus is public and the
README says so.

## 4. Components

One module, one job. Pure functions where possible so tests need no network.

| Module | Responsibility | Interface |
|---|---|---|
| `rag/settings.py` | Load `.env`: `GEMINI_API_KEY`, `GEMINI_BASE_URL`, model names, `DATABASE_URL`, chunk sizes, `RAG_MIN_SIMILARITY`, price table | `settings` object |
| `rag/providers.py` | One `openai` client against the Gemini endpoint, with rate limiting and retries. `embed` passes `dimensions=1536`. Chat returns real token usage; the endpoint returns none for embeddings, so `embed` estimates tokens as characters / 4 | `embed(texts) -> list[list[float]], usage` · `chat_json(system, user) -> dict, usage, latency_ms` |
| `rag/chunking.py` | Split page text into chunks of ~1800 characters with 200 overlap, breaking at paragraph or sentence boundaries when possible | `chunk_page(text) -> list[str]` (pure) |
| `rag/db.py` | Connection, schema creation, inserts and the two search queries | `init_schema()`, `upsert_document()`, `insert_chunks()`, `vector_search()`, `text_search()`, `log_query()` |
| `rag/ingest.py` | Read PDFs with pypdf page by page, chunk, embed in batches of 32, store. Skips unreadable files with a warning. Re-running replaces a document's chunks | `python -m rag.ingest [data/raw]` |
| `rag/retrieve.py` | Embed the question, run both searches, merge with reciprocal rank fusion (k = 60), return top-5 chunks plus the best cosine similarity | `retrieve(question) -> Retrieval(chunks, best_similarity)` · `rrf_merge(lists) -> ranked` (pure) |
| `rag/answer.py` | Build the numbered-sources prompt, call chat, validate the JSON, apply refusal rules | `answer(question) -> Answer(text, citations, refused, usage, cost_usd, latency_ms)` · `validate_citations()` (pure) |
| `rag/eval.py` | Run the golden set, compute metrics, write the results table | `python -m rag.eval` |
| `api.py` | FastAPI app: `POST /ask`, `GET /health` | HTTP |
| `scripts/fetch_corpus.py` | Download the eleven PDFs, report any failure, never overwrite an existing file | CLI |

## 5. Data model

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id          serial PRIMARY KEY,
  name        text UNIQUE NOT NULL,      -- file name without extension
  source_url  text,
  pages       int NOT NULL,
  ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
  id          serial PRIMARY KEY,
  document_id int NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page        int NOT NULL,              -- 1-based page in the PDF
  idx         int NOT NULL,              -- chunk order within the page
  text        text NOT NULL,
  embedding   vector(1536) NOT NULL,
  tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);

CREATE TABLE queries (
  id            serial PRIMARY KEY,
  asked_at      timestamptz NOT NULL DEFAULT now(),
  question      text NOT NULL,
  refused       boolean NOT NULL,
  top_documents text[] NOT NULL,
  cited_chunks  int[] NOT NULL,
  prompt_tokens int, completion_tokens int, embed_tokens int,
  cost_usd      numeric(10,6),
  latency_ms    int
);
```

`DATABASE_URL` defaults to `postgresql://rag:rag@localhost:5433/rag`; port 5433
so it never collides with another local PostgreSQL on 5432.

## 6. Retrieval

1. Embed the question (one Gemini embedding call, `dimensions=1536`).
2. Vector search: `ORDER BY embedding <=> $q LIMIT 20`, returning cosine
   similarity as `1 - distance`.
3. Full-text search: `WHERE tsv @@ websearch_to_tsquery('english', $q)` ordered
   by `ts_rank_cd`, limit 20.
4. Reciprocal rank fusion: `score(chunk) = Σ 1 / (60 + rank_in_list)` over the
   two lists; take the top 5.
5. `best_similarity` is the highest cosine similarity among the vector results
   and feeds the refusal rule. An empty result set (nothing ingested yet)
   counts as similarity 0, so the question is refused rather than answered
   from nothing.

Why hybrid: pure vector search is weak on exact terms such as "38 hours",
"Fair Work Information Statement" or clause names; full-text search catches
those, and RRF needs no weight tuning.

## 7. Answering and refusal

System prompt: answer only from the numbered sources, quote figures exactly,
cite by number, and return JSON `{"answer": str, "citations": [int],
"confidence": "high"|"medium"|"low"}`. If the sources do not contain the answer,
return an empty `citations` list and say so.

Refuse (reply "The documents I have do not cover this.") when any of:

- `best_similarity < RAG_MIN_SIMILARITY` (default 0.30; tuned on the golden set);
- the model returns no valid citation numbers;
- the JSON cannot be parsed after one retry with a stricter instruction (the
  response is also flagged `error=true` in the log).

Citations outside `1..5` are dropped; if none remain, the answer is refused.
Each kept citation is returned as document name, page and the chunk text.

## 8. API

`POST /ask` body `{"question": "..."}` → 

```json
{
  "answer": "...",
  "refused": false,
  "citations": [{"document": "Annual-leave", "page": 1, "snippet": "..."}],
  "latency_ms": 2140,
  "cost_usd": 0.000312
}
```

`GET /health` checks the database connection. Every `/ask` is logged to
`queries`. Errors: database down → 503; provider failure after retries → 502
with a short message; validation error → 422.

## 9. Evaluation

`evals/golden.yaml`:

```yaml
- id: annual-leave-weeks
  question: How many weeks of paid annual leave does a full-time employee get each year?
  expected_doc: Annual-leave
  gold_answer: 4 weeks
- id: off-topic-visa
  question: How do I apply for a 482 visa?
  expected_doc: null          # must refuse
```

Fifteen answerable questions and three off-topic ones. `python -m rag.eval`
runs each question through `retrieve` and `answer` and reports:

| Metric | Definition | Target |
|---|---|---|
| hit@5 | `expected_doc` is among the documents of the five retrieved chunks | ≥ 80% |
| cited hit | `expected_doc` is among the cited chunks' documents | reported |
| faithfulness | Gemini judge scores the answer 1–5 against the retrieved sources and the gold answer | mean ≥ 4 |
| refusal rate | share of off-topic questions refused | 3 of 3 |
| latency | mean and p95 ms per question | reported |
| cost | mean USD per question at list price | reported |

Results go to `evals/results/<UTC timestamp>.md` as a table; the README quotes
the latest run. The judge prompt lives in `rag/eval.py` and the judge uses the
same chat client (and therefore the same rate limiting).

## 10. Cost accounting

Price table in settings, USD per 1M tokens, marked approximate and overridable:
Gemini Embedding 2 0.20; Gemini 2.5 Flash input 0.30, output 2.50. Embedding
tokens are estimated (characters / 4) because the endpoint returns no usage for
embeddings. The free tier bills nothing, so the README states both the
list-price figure and that the actual spend was zero.

## 11. Error handling

- Provider 429/5xx (chat and embeddings): three retries, backoff 2 s, 4 s, 8 s; then the call fails
  and the caller reports it (eval marks the row, API returns 502).
- Unreadable PDF or empty page: skip with a warning, continue.
- JSON from the model: parse, validate shape, one retry, then refuse with
  `error=true`.
- Secrets only in `.env` (git-ignored); `.env.example` documents the keys.

## 12. Testing

`pytest` on pure functions, no network or database:

- `chunk_page`: respects the size limit, keeps the overlap, prefers paragraph
  breaks, never drops text.
- `rrf_merge`: a chunk present in both lists outranks one present in only one;
  ties keep input order.
- `validate_citations` and the refusal rule: out-of-range numbers dropped,
  empty citations refuse, low similarity refuses.

Acceptance for stage 1: eleven documents ingest and report chunk counts; the
golden set meets the targets above; `/ask` returns page-level citations.

## 13. Repository layout

```
au-workplace-assistant/
  README.md
  .env.example  .gitignore  requirements.txt  pyproject.toml
  docker-compose.yml          # pgvector/pgvector:pg16 on port 5433
  api.py
  rag/  settings.py providers.py chunking.py db.py ingest.py retrieve.py answer.py eval.py
  scripts/fetch_corpus.py
  evals/golden.yaml  evals/results/
  tests/  test_chunking.py test_retrieve.py test_answer.py
  data/raw/                   # downloaded PDFs, git-ignored
  docs/design/
```

Configuration (`.env`): `GEMINI_API_KEY`, `GEMINI_BASE_URL`, `CHAT_MODEL`,
`EMBED_MODEL`, `DATABASE_URL`, `RAG_MIN_SIMILARITY`, `CHAT_MIN_INTERVAL_S`.

## 14. Stage 2 hook

`retrieve()` and `answer()` are the first two tools of the later agent. The
agent lives in a new `agent/` package and does not change the modules above.
