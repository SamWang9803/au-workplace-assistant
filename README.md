# au-workplace-assistant

Grounded question answering over Australian workplace-entitlement fact sheets, with page-level citations, explicit refusal and a built-in evaluation harness.

English | [Chinese](README_CN.md)

> **Status:** stage 1 (retrieval, cited answers, evaluation) works end to end; see [Results](#results). A demo page at `GET /` was added on 2026-11-03. Stage 2 (an agent with tools and human-in-the-loop) is planned. Design notes live in [docs/design/](docs/design/), the build plan in [docs/tasks/](docs/tasks/).

## What it does

Ask a question about minimum workplace entitlements in Australia. The assistant answers only from eleven public Fair Work Ombudsman (FWO) fact sheets, names the document and page each answer came from, and refuses when the documents do not cover the question. A golden-set evaluation measures retrieval hit rate, faithfulness, refusal rate, latency and cost, so the quality claims in this README are measured rather than asserted.

Request:

```json
{"question": "How many weeks of annual leave does a full-time employee get?"}
```

Response from a real run, with the snippets shortened:

```json
{
  "answer": "A full-time employee is entitled to 4 weeks of paid annual leave each year. If the employee is classified as a shiftworker, they are entitled to 5 weeks of paid annual leave.",
  "refused": false,
  "confidence": "high",
  "citations": [
    {"document": "Annual-leave", "page": 1, "snippet": "..."},
    {"document": "Fair-Work-Information-Statement", "page": 2, "snippet": "..."},
    {"document": "Annual-leave", "page": 1, "snippet": "..."}
  ],
  "latency_ms": 2566,
  "cost_usd": 0.000955
}
```

The real response also carries the five retrieved `sources`, each with its similarity and whether it was cited; the demo page shows them under the answer.

This is a personal portfolio project. The pipeline is written by hand, without LangChain or LlamaIndex, so that every step is visible, unit-testable and explainable.

## Quick start

Needs Python 3.12, Docker Desktop and an OpenAI API key with a few dollars of credit. The first time takes about five minutes:

```powershell
# Windows PowerShell
docker compose up -d                                  # PostgreSQL 16 + pgvector on port 5433
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env                           # then put your key in OPENAI_API_KEY
.venv\Scripts\python scripts\fetch_corpus.py          # downloads the 11 PDFs into data\raw\
.venv\Scripts\python -m rag.ingest                    # chunks, embeds and stores them (about USD 0.001)
.venv\Scripts\python -m uvicorn api:app --reload      # then open http://127.0.0.1:8000/
```

```bash
# macOS / Linux
docker compose up -d
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                  # then put your key in OPENAI_API_KEY
python scripts/fetch_corpus.py
python -m rag.ingest
uvicorn api:app --reload                              # then open http://127.0.0.1:8000/
```

Every time after that:

```powershell
docker compose up -d
.venv\Scripts\python -m uvicorn api:app --reload      # macOS / Linux: source .venv/bin/activate && uvicorn api:app --reload
```

Stop with Ctrl+C, then `docker compose stop`. The Windows commands call the virtual environment's Python directly, so no activation script has to run. If `fetch_corpus.py` cannot download, save the eleven PDFs from the links under [Corpus](#corpus-attribution-and-disclaimer) into `data/raw/` by hand.

## Tech stack

| Layer | Choice |
|---|---|
| Language and API | Python 3.12, FastAPI |
| Storage | PostgreSQL 16 with pgvector in Docker: vectors, full-text index and query log in one database |
| Models | OpenAI `gpt-4.1-mini` for answers and the evaluation judge; `text-embedding-3-small` (1536 dimensions) for embeddings |
| PDF text | pypdf |
| Tests | pytest; no network or database needed, and real OpenAI calls are blocked |

## How it works, in plain language

Think of the assistant as a librarian who may only answer from eleven booklets and has to point to the page every time.

**Preparing the shelf (ingestion, run once).** Each PDF is cut into cards of about 1800 characters, and every card is labelled with its document and page. Every card also gets an embedding: a list of 1536 numbers that works like coordinates on a map of meaning, so cards about similar things sit close together. Card text, coordinates and a keyword index all go into one PostgreSQL database.

**Answering a question (every request).**

1. The question gets coordinates too.
2. Two searches run side by side. The vector search picks the 20 cards closest in meaning; the full-text search picks the 20 cards that share the question's keywords. Meaning-based search misses exact tokens such as "38 hours", keyword search misses paraphrase, so each covers the other's blind spot.
3. Reciprocal rank fusion merges the two lists into one. A card scores 1 / (60 + rank) in each list it appears in, so cards high on both lists win. The top five survive.
4. The five cards are numbered [1] to [5] and handed to the model with one rule: answer only from these cards, copy figures exactly, say which cards you used. The reply has to follow a fixed JSON shape.
5. The server checks the homework. Citation numbers outside 1 to 5 are dropped. If the best card scored below the similarity threshold, or the model cited nothing, the answer becomes "The documents I have do not cover this."
6. Citation numbers are translated back into document, page and snippet, and the request is logged with its tokens, cost and latency.

**Example.** "How many weeks of paid annual leave does a full-time employee get?" The annual-leave card ranks first, well above the threshold; the model answers "4 weeks" citing [1]; the response points at Annual-leave, page 1. "What is the capital gains tax discount on shares?" finds no card above the threshold, so the system refuses without calling the model at all. "How do I apply for a 482 visa?" does get past the threshold, because work visas sit close to workplace topics, but no card answers it, the model cites none, and the answer is refused too.

## Architecture

```mermaid
flowchart LR
  subgraph Ingest["Ingest: python -m rag.ingest"]
    PDF["11 FWO PDFs<br/>data/raw/"] --> PP["pypdf<br/>text per page"]
    PP --> CH["chunk_page<br/>~1800 chars, 200 overlap"]
    CH --> EM1["text-embedding-3-small<br/>batches of 64"]
    EM1 --> DB[("PostgreSQL 16 + pgvector<br/>chunks: text, embedding, tsvector")]
  end
  subgraph Query["Query: POST /ask"]
    Q["question"] --> EM2["embed question"]
    EM2 --> VS["vector search<br/>top-20 by cosine"]
    Q --> TS["full-text search<br/>any of the question's words, top-20"]
    VS --> RRF["reciprocal rank fusion<br/>top-5"]
    TS --> RRF
    RRF --> PR["prompt with<br/>numbered sources"]
    PR --> LLM["gpt-4.1-mini<br/>strict JSON schema"]
    LLM --> VAL["validate citations<br/>apply refusal rules"]
    VAL --> A["answer + citations<br/>document, page, snippet"]
    VAL --> LOG[("queries table<br/>tokens, cost, latency")]
  end
  DB -.-> VS
  DB -.-> TS
```

Two flows share one database. Ingestion runs once, and again whenever a PDF changes. Querying runs per request.

### Components

One module, one job. Pure functions wherever possible, so unit tests need neither network nor database.

| Module | Responsibility | Interface |
|---|---|---|
| `rag/settings.py` | The settings panel. On start-up it reads `.env` and gathers the OpenAI key, the chat and embedding model names, the database URL and the similarity threshold into one `settings` object, next to fixed constants for the chunk sizes and the price per 1M tokens. It also holds the list of the eleven corpus URLs, shared by the download script and ingestion. Every other module takes its configuration from here instead of reading files itself | `settings` object · `CORPUS_URLS` |
| `rag/providers.py` | The only place in the project that talks to OpenAI. `embed` takes a batch of texts and returns each text's 1536-number coordinates plus the tokens used; `chat_json` takes a system prompt, a user prompt and a JSON schema, makes the model answer in that shape, and returns the parsed reply (None if the model produced no usable JSON), token usage and latency; `cost_usd` turns token counts into dollars. The SDK retries 429 rate limits and 5xx errors three times with exponential backoff. No other module imports `openai` | `embed(texts) -> (vectors, tokens)` · `chat_json(system, user, schema) -> ChatReply(data, prompt_tokens, completion_tokens, latency_ms)` · `cost_usd(...)` |
| `rag/chunking.py` | Cuts the cards. `clean_text` first tidies what pypdf extracted (NUL bytes, repeated spaces, runs of blank lines). `chunk_page` then splits one page into chunks of about 1800 characters with up to 200 characters of overlap, breaking at blank lines (paragraphs) first, then at sentence ends, then with a hard cut. Neither touches network nor database, and the same input always gives the same output, so pytest can check "never too long, overlap present, no text lost" | `clean_text(text) -> str` · `chunk_page(text) -> list[str]` (both pure) |
| `rag/db.py` | Home of all SQL. Connects to the database, creates tables and indexes, inserts documents and chunks, finds the nearest chunks by coordinates (vector search), finds chunks that share words with the question (full-text search) and writes each request to the log table. Other modules call these functions instead of writing SQL; each takes an open connection, so ingestion can wrap a document's writes in one transaction | `connect()`, `init_schema()`, `upsert_document()`, `insert_chunks()`, `vector_search()`, `text_search()`, `log_query()` |
| `rag/ingest.py` | Conductor of the ingestion flow. Walks every PDF in `data/raw/`, pulls text page by page with pypdf, hands pages to `chunking`, sends chunks to `providers.embed` 64 at a time and stores everything through `db`. Unreadable files are skipped with a warning. Re-running deletes a document's old chunks before writing new ones, so the table never grows by repetition. Prints pages and chunks per document | `python -m rag.ingest [data/raw]` |
| `rag/retrieve.py` | First half of answering: finding the material. Gets the question's coordinates from `providers.embed`, asks `db` for the top 20 by vector and the top 20 by full text, merges both lists with `rrf_merge` and keeps five, and returns them with the best cosine similarity from the vector list. `rrf_merge` is a pure function with its own tests | `retrieve(question) -> Retrieval(chunks, best_similarity, embed_tokens)` · `rrf_merge(lists) -> ranked` (pure) |
| `rag/answer.py` | Second half of answering: writing the answer and checking it. Numbers the five chunks from `retrieve` into the prompt, calls `providers.chat_json` for the JSON reply, drops out-of-range citations with `validate_citations`, then applies the refusal rules: similarity too low, no valid citation or unparseable JSON means refuse. Finally computes the cost from token usage and returns everything together | `answer(question) -> Answer(text, citations, refused, error, sources, tokens, cost_usd, latency_ms)` · `validate_citations()` (pure) |
| `rag/eval.py` | The exam. Reads the 18 questions in `evals/golden.yaml`, runs each through `retrieve` and `answer`, checks whether the expected document was retrieved and cited and whether off-topic questions were refused, asks the model to judge each answer's faithfulness from 1 to 5, aggregates hit rate, mean score, latency and cost, and writes a Markdown table to `evals/results/`. With `--retrieval-only` it skips the chat model and reports hit@5 and similarities for a fraction of a cent | `python -m rag.eval [--retrieval-only]` |
| `api.py` | The network front door. A FastAPI app with two routes: `POST /ask` takes a question, calls `answer`, returns the answer and citations as JSON and logs the request; `GET /health` checks the database connection. Database down gives 503, provider failure after retries 502, a malformed request 422 | `POST /ask`, `GET /health` |
| `scripts/fetch_corpus.py` | The download script. Fetches the eleven PDFs listed in `rag/settings.py` from fairwork.gov.au into `data/raw/`, skips files that already exist instead of overwriting them, prints one success or failure line per file and exits non-zero if anything failed | CLI |

### 1. Ingestion

1. `scripts/fetch_corpus.py` downloads the eleven PDFs into `data/raw/`. The folder is git-ignored; the PDFs are not redistributed.
2. `rag/ingest.py` extracts text page by page with pypdf and tidies it with `clean_text`. Unreadable files and empty pages are skipped with a warning.
3. `chunk_page` splits each page into chunks of about 1800 characters with 200 characters of overlap. It breaks at paragraph boundaries first, then at sentence ends, then with a hard cut. No text is dropped. Chunks never cross a page, so every chunk carries a page number for citations.
4. Chunks are embedded with `text-embedding-3-small` in batches of 64 and stored in the `chunks` table next to a generated `tsvector` for full-text search.
5. Re-ingesting a document deletes its old rows first, so running ingest twice never duplicates data.

### 2. Retrieval: hybrid search with reciprocal rank fusion

1. Embed the question (one embedding call).
2. Vector search: `ORDER BY embedding <=> $q LIMIT 20`, with cosine similarity reported as `1 - distance`.
3. Full-text search: the question's words are stemmed with `plainto_tsquery('english', $q)` and joined with OR, so a chunk needs only some of them; matches are ranked by `ts_rank_cd`, limit 20. Joined with AND, as `plainto_tsquery` and `websearch_to_tsquery` do by default, a whole question matches almost nothing: "How many weeks of annual leave?" would require the word "many" in the chunk.
4. Reciprocal rank fusion: `score(chunk) = Σ 1 / (60 + rank_in_list)` across both lists; keep the top 5.
5. `best_similarity`, the highest cosine similarity among the vector results, feeds the refusal rule. An empty index counts as similarity 0, so a system with nothing ingested refuses instead of answering from nothing.

Why hybrid: embeddings alone are weak on exact tokens such as "38 hours", "Fair Work Information Statement" or clause names. Full-text search catches those, and RRF merges the two rankings without any score normalisation or weight tuning.

### 3. Answering and refusal

The prompt lists the five retrieved chunks as numbered sources. The system prompt instructs the model to answer only from those sources, quote figures exactly, cite by source number, and return JSON of this shape:

```json
{"answer": "string", "citations": [1, 3], "confidence": "high | medium | low"}
```

A strict `json_schema` response format enforces the shape. Temperature is 0 so evaluation runs are comparable.

The assistant refuses, replying "The documents I have do not cover this.", when any of these holds:

- `best_similarity < RAG_MIN_SIMILARITY` (default 0.30, tuned on the golden set);
- the model returns no valid citation numbers;
- the model returns no usable JSON. The strict schema leaves only a safety refusal or a cut-off reply as causes, which a retry would not fix, so the query is refused at once and flagged `error=true` in the log.

The similarity check runs before the model is called, so off-topic questions cost no chat tokens. Citation numbers outside `1..5` are dropped; if none remain, the answer is refused. Each kept citation is returned as document name, page and chunk text, so a reader can check the claim against the original PDF.

### 4. Evaluation

`evals/golden.yaml` holds fifteen answerable questions spread across the eleven documents and three off-topic questions that must be refused:

```yaml
- id: annual-leave-weeks
  question: How many weeks of paid annual leave does a full-time employee get each year?
  expected_doc: Annual-leave
  gold_answer: 4 weeks
- id: off-topic-visa
  question: How do I apply for a 482 visa?
  expected_doc: null          # must refuse
```

`python -m rag.eval` runs every question through `retrieve` and `answer` and reports:

| Metric | Definition | Target |
|---|---|---|
| hit@5 | `expected_doc` is among the documents of the five retrieved chunks | ≥ 80% |
| cited hit | `expected_doc` is among the documents actually cited | reported |
| faithfulness | an LLM judge (the same chat model) scores the answer 1–5 against the retrieved sources and the gold answer | mean ≥ 4 |
| refusal rate | share of off-topic questions refused | 3 of 3 |
| latency | mean and p95 milliseconds per question | reported |
| cost | mean USD per question at list price | reported |

Each run writes a Markdown table to `evals/results/<UTC timestamp>.md`. Those files are committed, so the numbers quoted in this README can be reproduced.

`python -m rag.eval --retrieval-only` runs only the retrieval half: hit@5 and each question's best similarity, for a fraction of a cent. Use it to tune chunking and `RAG_MIN_SIMILARITY` before paying for a full run with the judge.

### 5. Cost and latency accounting

Every `/ask` writes one row to `queries`: the question, whether it was refused, the top documents, the cited chunks, prompt, completion and embedding tokens, cost in USD and latency. Cost is computed from the token usage the API returns and a price table in settings (USD per 1M tokens, marked approximate and overridable): embeddings 0.02; `gpt-4.1-mini` input 0.40, output 1.60. A full evaluation run cost USD 0.029 at these prices.

### 6. Data model

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
  error         boolean NOT NULL DEFAULT false,  -- the model returned no usable JSON
  top_documents text[] NOT NULL,
  cited_chunks  int[] NOT NULL,
  prompt_tokens int, completion_tokens int, embed_tokens int,
  cost_usd      numeric(10,6),
  latency_ms    int
);
```

`DATABASE_URL` defaults to `postgresql://rag:rag@127.0.0.1:5433/rag`. Port 5433 avoids clashing with a local PostgreSQL on 5432. The address is `127.0.0.1` rather than `localhost` because on Windows `localhost` tries IPv6 first, and against Docker's IPv4-only port that stalled every connection for minutes.

### 7. HTTP API

| Endpoint | Behaviour |
|---|---|
| `GET /` | The demo page: a question form with example chips, the answer with an Answered or Refused badge, citation cards, and the five retrieved sources in a collapsible table |
| `POST /ask` with `{"question": "..."}` | Returns `answer`, `refused`, `confidence`, `citations[]` (rank, document, page, snippet), `sources[]` (the fused top-5: rank, document, page, snippet, similarity or null for full-text-only hits, cited), `latency_ms`, `cost_usd`, and logs the query |
| `GET /health` | Checks the database connection |

Error mapping: database down → 503; model provider failure after retries → 502 with a short message; invalid request body, such as an empty question or one over 1000 characters → 422. The OpenAI SDK retries 429 and 5xx responses three times with exponential backoff; a database connection gives up after 5 seconds.

## Running everything

After the quick start, with the virtual environment's Python (`.venv\Scripts\python` on Windows; `python` after `source .venv/bin/activate` elsewhere):

```bash
# Ask from the command line (each answer costs about USD 0.001)
python -m rag.retrieve "How many hours is a full-time week?"      # retrieval only, no model call
python -m rag.answer "How many weeks of paid annual leave does a full-time employee get?"
python -m rag.answer "How do I apply for a 482 visa?"             # refused

# Call the API while uvicorn is running
curl -X POST 127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How much notice does an employer have to give?"}'

# Tests and evaluation
pytest                               # no network: OpenAI calls are blocked in tests
python -m rag.eval --retrieval-only  # retrieval metrics only, a fraction of a cent
python -m rag.eval                   # full run with the judge, about USD 0.03
```

In Windows PowerShell 5.1, `curl` is an alias of `Invoke-WebRequest` with different flags, so call the API like this instead:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" -Body '{"question": "How much notice does an employer have to give?"}' | ConvertTo-Json -Depth 5
```

Unit tests cover the pure functions: `chunk_page` (size limit, overlap, paragraph preference, no text lost), `rrf_merge` (a chunk in both lists outranks one in a single list, ties keep input order) and `validate_citations`. The refusal rules are tested through `answer()` itself, with retrieval and the model replaced by fakes: low similarity refuses without calling the model, empty or out-of-range citations refuse, unusable JSON refuses with `error=true`. A fixture blocks every real OpenAI call, so `pytest` never spends money. `tests/test_api.py` drives the HTTP layer with FastAPI's test client and a fake pipeline: the demo page is served, `/ask` returns the documented fields with matching `cited` flags, invalid questions give 422.

## Configuration

All settings live in `.env`, which is git-ignored. `.env.example` documents them.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | required | Answers, the evaluation judge and embeddings |
| `CHAT_MODEL` | `gpt-4.1-mini` | Answer and judge model |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model; its 1536 dimensions must match `vector(1536)` in the schema |
| `DATABASE_URL` | `postgresql://rag:rag@127.0.0.1:5433/rag` | PostgreSQL connection string |
| `RAG_MIN_SIMILARITY` | `0.30` | Refuse when the best cosine similarity is below this |

## Results

Latest run: [evals/results/2026-11-02T10-29-04Z.md](evals/results/2026-11-02T10-29-04Z.md), with `gpt-4.1-mini`, `text-embedding-3-small` and `RAG_MIN_SIMILARITY` 0.30 over the 18 golden questions.

| Metric | Target | Latest run |
|---|---|---|
| hit@5 | ≥ 80% | 15/15 (100%) |
| cited hit | reported | 15/15 (100%) |
| faithfulness (mean, 1–5) | ≥ 4 | 4.80 over 15 answered |
| refusal rate (off-topic) | 3 of 3 | 3/3 |
| answerable questions refused | reported | 0 |
| latency (mean / p95) | reported | 1314 / 2678 ms |
| cost per question (list price) | reported | USD 0.00083 |

The whole run cost USD 0.029 at list price: 0.015 for answering and 0.014 for the judge. Two off-topic questions scored about 0.25 and were refused by the similarity threshold without calling the model. The third, about a 482 visa, scored 0.34, passed the threshold, and was refused because the model cited no source.

**One wrong answer, and why.** The redundancy question scored 2: the system said 4 weeks' pay for 5 years of service, where the fact sheet says 10 weeks. Retrieval was right, and the correct chunk was cited; the text inside it was not. In that table each row's range wraps onto two lines, and pypdf emits the second line first ("At least / 6 years / 5 years but less than 10 weeks"), so the model read scrambled rows. pypdf's layout mode keeps the rows in order but interleaves the page's two text columns, so it is not a drop-in fix; a table-aware extractor is listed under next steps.

Total spent building stage 1, computed from the token usage the API reported: about USD 0.03.

## Design decisions and trade-offs

- **No orchestration framework.** LangChain or LlamaIndex would save a few hundred lines, but they hide the steps that matter here: how text is chunked, how the two rankings are merged, when the system refuses. Writing those by hand keeps each one a small, unit-tested function. The cost is that swapping a component means writing code rather than changing a config.
- **PostgreSQL + pgvector instead of a dedicated vector database.** One container holds the vectors, the full-text index and the query log, so hybrid search is a single SQL round-trip and operations stay familiar. HNSW on the `vector` type is capped at 2000 dimensions and will not match a purpose-built store at tens of millions of vectors. Neither matters for a few hundred chunks.
- **Chunks of about 1800 characters with 200 overlap, split on paragraph then sentence boundaries.** Large enough that a fact and its qualifier usually stay together, small enough that five chunks fit comfortably in the prompt. The overlap stops a sentence that straddles a boundary from being lost. Chunks never cross a page, which keeps citations page-accurate at the price of occasionally splitting a table.
- **Hybrid retrieval merged with reciprocal rank fusion (k = 60).** Vector search handles paraphrase; full-text search handles exact terms and numbers. RRF needs no score normalisation or weight tuning, which matters when the tuning set has eighteen questions. Full-text matching ORs the question's words, because AND semantics demand that every word of a natural question appear in one chunk. Fixed top-20 and top-5 cut-offs are simple but not adaptive.
- **Refusal on a similarity threshold plus citation validation.** Two independent signals: a retrieval-side check (best cosine similarity below 0.30) and a generation-side check (no valid citation). On the golden set, answerable questions scored at least 0.53 and off-topic ones at most 0.34, so 0.30 leaves room on the answerable side; the visa question, which slipped past it, shows why the citation check is needed as well. The threshold is still the most fragile parameter in the system, and a different embedding model would need it re-tuned.
- **Citations are source numbers, mapped back on the server.** The model only ever emits numbers `1..5`; the server maps them to real chunks, pages and text. The model cannot invent a document or a page, and out-of-range numbers are dropped.
- **Strict JSON schema and temperature 0.** Structured outputs guarantee a parseable, correctly shaped response, so there is no parse-and-retry loop, and temperature 0 makes evaluation runs comparable. Deterministic sampling is not a deterministic model, so small run-to-run differences still occur.
- **`gpt-4.1-mini` rather than the `gpt-5` family.** The `gpt-5` models accept only the default temperature, which would make the evaluation noisier. `gpt-4.1-mini` is cheap (USD 0.40 input, 1.60 output per 1M tokens) and strong enough to answer from five short sources.
- **One provider for everything.** The first design paired Gemini's free tier for chat with OpenAI embeddings. The free tier on this project allowed 20 chat requests per day, while one evaluation run needs 33 calls, so chat moved to OpenAI as well. One key, one client, about USD 0.03 per evaluation run.
- **LLM-as-judge against a gold answer.** The judge sees the retrieved sources and the expected answer, which keeps the 1–5 scores anchored. Using the same model as answerer and judge invites self-preference bias. The retrieval metrics (hit@5, cited hit) do not depend on the judge, and production would use a different judge model with human spot checks.
- **A small golden set.** Eighteen questions catch regressions and keep every run cheap, but they are not a statistically strong measure of quality.
- **A static demo page instead of a front-end framework.** One HTML file with inline CSS and JavaScript, served by FastAPI, calling the same `/ask` the evaluation uses. It shows the retrieved sources next to the citations so hybrid retrieval and refusal can be demonstrated live. No history, streaming or login; see [docs/design/2026-11-03-demo-page-design.md](docs/design/2026-11-03-demo-page-design.md).

## What production would need next

Deliberately out of scope for stage 1:

- a real front end (the demo page has no history, streaming or login), authentication and multi-tenancy;
- a table-aware PDF extractor, because pypdf scrambles table cells that wrap onto two lines (see Results);
- a reranker over the fused top-k, and caching of embeddings and answers;
- conversation memory for follow-up questions;
- a different, stronger model as the evaluation judge, plus human spot checks;
- a document-refresh pipeline, because FWO fact sheets change (minimum wages, for example, change every 1 July);
- monitoring and alerting on latency, cost and refusal rate;
- a larger, versioned golden set.

## Roadmap: stage 2

`retrieve()` and `answer()` become the first two tools of an agent in a new `agent/` package, with tool use, human-in-the-loop approval and an audit log. The stage 1 modules do not change.

## Repository layout

```
au-workplace-assistant/
  README.md  README_CN.md
  .env.example  .gitignore  requirements.txt  pyproject.toml
  docker-compose.yml          # pgvector/pgvector:pg16 on port 5433
  api.py
  static/index.html           # demo page, served at GET /
  rag/  settings.py providers.py chunking.py db.py ingest.py retrieve.py answer.py eval.py
  scripts/fetch_corpus.py
  evals/golden.yaml  evals/results/
  tests/  test_chunking.py test_retrieve.py test_answer.py test_api.py
  data/raw/                   # downloaded PDFs, git-ignored
  docs/design/                # design notes (English and Chinese)
  docs/tasks/                 # implementation plan
```

## Corpus, attribution and disclaimer

The corpus is eleven FWO fact sheets, downloaded at build time from `https://www.fairwork.gov.au/sites/default/files/migration/`:

| Document | Path |
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

Source material is © Commonwealth of Australia (Fair Work Ombudsman). The repository stores only derived chunks locally and does not redistribute the PDFs. Any additional `.pdf` placed in `data/raw/` is ingested the same way.

This project is a technical demonstration, not legal advice. Entitlements change over time; check [fairwork.gov.au](https://www.fairwork.gov.au) for current figures. OpenAI API data is not used for model training by default.

## License

Apache License 2.0. See [LICENSE](LICENSE).
