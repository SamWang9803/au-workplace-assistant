# au-workplace-assistant

基于澳大利亚职场权益说明文档的有据问答：回答精确引用到页，文档没覆盖就明确拒答，并自带评测。

[English](README.md) | 中文

> **状态：** 第一阶段（检索、带引用的回答、评测）已经端到端跑通，见[评测结果](#评测结果)；2026-11-03 加了演示页 `GET /`；第二阶段（带工具和人工介入的 agent）在计划中。设计稿见 [docs/design/](docs/design/)，实施计划见 [docs/tasks/](docs/tasks/)。

## 它做什么

问一个关于澳大利亚最低职场权益的问题，助手只根据 Fair Work Ombudsman（FWO，公平工作监察署）公开的 11 份说明文档作答，每个回答都注明出自哪份文档的第几页；文档没有覆盖的问题，它会拒答而不是猜。一套黄金测试集会测出检索命中率、忠实度、拒答率、延迟和费用，所以本文里的质量说法是测出来的，不是口头宣称的。

请求：

```json
{"question": "How many weeks of annual leave does a full-time employee get?"}
```

响应（一次真实运行的结果，片段已省略）：

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

真实响应里还带着检索到的 5 个 `sources`，各有相似度和是否被引用；演示页会把它们列在回答下面。

这是一个个人作品集项目。流水线全部手写，不用 LangChain 或 LlamaIndex，为的是每一步都看得见、能做单元测试、能讲清楚。

## 快速上手

需要 Python 3.12、Docker Desktop，以及一把有几美元额度的 OpenAI API key。第一次大约五分钟：

```powershell
# Windows PowerShell
docker compose up -d                                  # PostgreSQL 16 + pgvector，端口 5433
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env                           # 然后把 key 填进 OPENAI_API_KEY
.venv\Scripts\python scripts\fetch_corpus.py          # 把 11 份 PDF 下载到 data\raw\
.venv\Scripts\python -m rag.ingest                    # 分块、嵌入、入库（约 0.001 美元）
.venv\Scripts\python -m uvicorn api:app --reload      # 然后打开 http://127.0.0.1:8000/
```

```bash
# macOS / Linux
docker compose up -d
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                  # 然后把 key 填进 OPENAI_API_KEY
python scripts/fetch_corpus.py
python -m rag.ingest
uvicorn api:app --reload                              # 然后打开 http://127.0.0.1:8000/
```

以后每次只需要：

```powershell
docker compose up -d
.venv\Scripts\python -m uvicorn api:app --reload      # macOS / Linux: source .venv/bin/activate && uvicorn api:app --reload
```

停止时先按 Ctrl+C，再执行 `docker compose stop`。Windows 的命令直接调用虚拟环境里的 Python，不需要运行激活脚本。如果 `fetch_corpus.py` 下载不了，就按[语料](#语料版权与免责声明)一节的链接把 11 份 PDF 手动存进 `data/raw/`。

## 技术栈

| 层 | 选择 |
|---|---|
| 语言与 API | Python 3.12，FastAPI |
| 存储 | Docker 里的 PostgreSQL 16 + pgvector：向量、全文索引和查询日志都在同一个库 |
| 模型 | OpenAI `gpt-4.1-mini` 负责回答和评测评判；`text-embedding-3-small`（1536 维）负责向量嵌入 |
| PDF 取文本 | pypdf |
| 测试 | pytest；不需要联网也不需要数据库，真实的 OpenAI 调用会被拦下 |

## 工作原理（通俗版）

把这个助手想成一位图书管理员：他只准根据 11 本小册子答题，而且每次都必须指出出自哪本、第几页。

**备课（入库，只做一次）。** 每份 PDF 被切成约 1800 字符一张的卡片，每张卡片标着文档名和页码。每张卡片再算一个 embedding：由 1536 个数字组成的一串，相当于「意思地图」上的坐标，讲相近内容的卡片坐标也相近。卡片文字、坐标和一份关键词索引一起存进同一个 PostgreSQL 数据库。

**答题（每次提问）。**

1. 问题也算出坐标。
2. 两路搜索同时进行。向量检索挑出意思最接近的 20 张卡片；全文检索挑出和问题共享关键词的 20 张。按意思找会漏掉「38 hours」这种精确词，按关键词找又不懂同义改写，两路互相补盲区。
3. 倒数排名融合（RRF）把两份榜单合成一份。卡片在每份榜单里得 1 / (60 + 名次) 分，两边都靠前的卡片得分最高。留下前 5 张。
4. 这 5 张卡片编号 [1] 到 [5]，连同一条规则交给模型：只准根据这几张卡片作答，数字原样抄，说明用了哪几张。回复必须符合固定的 JSON 结构。
5. 服务端检查作业。编号不在 1 到 5 内的引用丢弃。如果最像的卡片相似度低于阈值，或模型一张都没引用，回答就变成「The documents I have do not cover this.」。
6. 引用编号翻译回文档名、页码和原文片段；这次请求连同 token 数、费用和耗时一起记入日志。

**举例。** 问「全职员工每年有几周带薪年假」：年假那本的卡片排第一，远高于相似度阈值；模型回答「4 weeks」并引用 [1]；响应里指向 Annual-leave 第 1 页。问「持有股票的资本利得税折扣是多少」：没有任何卡片超过阈值，系统直接拒答，根本不会调用模型。问「怎么申请 482 签证」：工作签证和职场话题离得近，相似度过了阈值，但没有卡片能回答它，模型一条都没引用，所以同样被拒答。

## 架构

```mermaid
flowchart LR
  subgraph Ingest["入库: python -m rag.ingest"]
    PDF["11 份 FWO PDF<br/>data/raw/"] --> PP["pypdf<br/>逐页取文本"]
    PP --> CH["chunk_page<br/>约 1800 字符, 重叠 200"]
    CH --> EM1["text-embedding-3-small<br/>每批 64 条"]
    EM1 --> DB[("PostgreSQL 16 + pgvector<br/>chunks: text, embedding, tsvector")]
  end
  subgraph Query["查询: POST /ask"]
    Q["问题"] --> EM2["问题做嵌入"]
    EM2 --> VS["向量检索<br/>按余弦取 top-20"]
    Q --> TS["全文检索<br/>任一问题词匹配, top-20"]
    VS --> RRF["倒数排名融合 RRF<br/>取 top-5"]
    TS --> RRF
    RRF --> PR["带编号来源的 prompt"]
    PR --> LLM["gpt-4.1-mini<br/>严格 JSON schema"]
    LLM --> VAL["校验引用<br/>套用拒答规则"]
    VAL --> A["回答 + 引用<br/>文档、页码、片段"]
    VAL --> LOG[("queries 表<br/>token、费用、延迟")]
  end
  DB -.-> VS
  DB -.-> TS
```

两条流程共用一个数据库。入库跑一次，PDF 有变化时再跑；查询按请求逐次执行。

### 组件

一个模块只干一件事。能写成纯函数的尽量写成纯函数，这样单元测试既不用联网也不用连库。

| 模块 | 职责 | 接口 |
|---|---|---|
| `rag/settings.py` | 程序的「设置面板」。启动时读取 `.env`，把 OpenAI 的 key、用哪个对话模型和嵌入模型、数据库地址、相似度阈值多少，集中放进一个 `settings` 对象；卡片切多大、每百万 token 多少钱是写在这里的常量。11 份语料的下载地址清单也放在这里，下载脚本和入库共用。其他模块都从这里取配置，自己不读文件 | `settings` 对象 · `CORPUS_URLS` |
| `rag/providers.py` | 整个项目里唯一和 OpenAI 说话的地方。`embed` 收一批文字，返回每段文字的 1536 维坐标和用掉的 token 数；`chat_json` 收系统提示、用户提示和一份 JSON 结构说明，让模型按这个结构作答，返回解析好的回复（模型没给出可用 JSON 时为 None）、token 用量和耗时；`cost_usd` 把 token 数换算成美元。遇到 429 限流或 5xx 故障，由 SDK 按指数退避自动重试三次。别的模块不直接 import `openai` | `embed(texts) -> (vectors, tokens)` · `chat_json(system, user, schema) -> ChatReply(data, prompt_tokens, completion_tokens, latency_ms)` · `cost_usd(...)` |
| `rag/chunking.py` | 切卡片。先用 `clean_text` 清理 pypdf 抽出的文字（去掉 NUL 字符、多余空格和连续空行），再由 `chunk_page` 把一页切成约 1800 字符一块，相邻块最多重叠 200 字符；优先在空行（段落）处断，其次在句号处断，实在不行才硬切。两者都不碰网络和数据库，输入相同输出必相同，所以能用 pytest 反复验证「不超长、有重叠、不丢字」 | `clean_text(text) -> str` · `chunk_page(text) -> list[str]`（都是纯函数） |
| `rag/db.py` | 所有 SQL 的家。负责连数据库、建表建索引、写入文档和卡片、按坐标找最近的卡片（向量检索）、找和问题有共同词的卡片（全文检索）、把每次提问记进日志表。别的模块不写 SQL，只调这里的函数；每个函数都接收一个已打开的连接，这样入库时能把一份文档的写入包进同一个事务 | `connect()`、`init_schema()`、`upsert_document()`、`insert_chunks()`、`vector_search()`、`text_search()`、`log_query()` |
| `rag/ingest.py` | 备课流程的总指挥。扫描 `data/raw/` 里的每个 PDF，用 pypdf 逐页取文字，交给 `chunking` 切卡片，攒够 64 张就交给 `providers.embed` 算坐标，再通过 `db` 存库。读不了的文件打警告跳过。重跑时先删掉该文档旧的卡片再写新的，所以不会越跑越多。每处理完一份就打印页数和卡片数 | `python -m rag.ingest [data/raw]` |
| `rag/retrieve.py` | 答题流程的前半段：找资料。把问题交给 `providers.embed` 算坐标，调 `db` 做向量检索和全文检索各取 20 张，用 `rrf_merge` 合成一份榜单取前 5，连同向量检索里的最高相似度一起返回。`rrf_merge` 是纯函数，可以单独测试 | `retrieve(question) -> Retrieval(chunks, best_similarity, embed_tokens)` · `rrf_merge(lists) -> ranked`（纯函数） |
| `rag/answer.py` | 答题流程的后半段：写答案并自查。把 `retrieve` 给的 5 张卡片编号拼进 prompt，调 `providers.chat_json` 拿到 JSON 回答，用 `validate_citations` 丢掉越界编号，再按拒答规则判断：相似度太低、没有有效引用、JSON 解析失败都拒答。最后按 token 用量算出这次花了多少钱，一起返回 | `answer(question) -> Answer(text, citations, refused, error, sources, tokens, cost_usd, latency_ms)` · `validate_citations()`（纯函数） |
| `rag/eval.py` | 考试系统。读 `evals/golden.yaml` 里的 18 道题，每题走一遍 `retrieve` 和 `answer`，核对检出的文档对不对、引用对不对、该拒答的有没有拒答，再让模型当评判给每个回答的忠实度打 1 到 5 分，汇总命中率、平均分、延迟和费用，写成一张 Markdown 表存进 `evals/results/`。加 `--retrieval-only` 时不调对话模型，只报 hit@5 和相似度，花费不到一美分 | `python -m rag.eval [--retrieval-only]` |
| `api.py` | 对外的网络入口。用 FastAPI 开两个接口：`POST /ask` 收问题、调 `answer`、把回答和引用以 JSON 返回并记日志；`GET /health` 检查数据库连不连得上。数据库挂了返回 503，模型服务重试后仍失败返回 502，请求格式不对返回 422 | `POST /ask`、`GET /health` |
| `scripts/fetch_corpus.py` | 下载脚本。按 `rag/settings.py` 里的清单从 fairwork.gov.au 把 11 份 PDF 下到 `data/raw/`，已经存在的文件跳过不覆盖，每个文件打印一行成功或失败，有任何失败就以非零退出码结束 | 命令行 |

### 1. 入库

1. `scripts/fetch_corpus.py` 把 11 份 PDF 下载到 `data/raw/`。该目录在 .gitignore 里，PDF 不随仓库分发。
2. `rag/ingest.py` 用 pypdf 逐页取文本，再用 `clean_text` 清理。读不了的文件和空白页打警告跳过。
3. `chunk_page` 把每页切成约 1800 字符、重叠 200 字符的块：优先在段落边界断开，其次是句末，最后才硬切。不丢任何文本。分块不跨页，所以每个分块都带页码，引用能精确到页。
4. 分块用 `text-embedding-3-small` 按每批 64 条做嵌入，和一个自动生成的 `tsvector`（供全文检索）一起存进 `chunks` 表。
5. 重新入库某份文档时先删掉它的旧行，所以入库跑两遍也不会产生重复数据。

### 2. 检索：混合检索 + 倒数排名融合

1. 把问题做嵌入（一次 embedding 调用）。
2. 向量检索：`ORDER BY embedding <=> $q LIMIT 20`，余弦相似度按 `1 - distance` 报告。
3. 全文检索：用 `plainto_tsquery('english', $q)` 把问题里的词做词干化，再用 OR 连起来，一张卡片只要含有其中一部分词就算命中；命中结果按 `ts_rank_cd` 排序，取 20 条。如果像 `plainto_tsquery` 和 `websearch_to_tsquery` 默认那样用 AND 连接，一整句问题几乎什么都匹配不到：问「How many weeks of annual leave?」会要求卡片里出现「many」这个词。
4. 倒数排名融合（RRF）：对两个列表求 `score(chunk) = Σ 1 / (60 + rank_in_list)`，取前 5。
5. `best_similarity` 是向量检索结果里的最高余弦相似度，供拒答规则使用。索引为空时按相似度 0 处理，所以什么都没入库的系统会拒答，而不是凭空作答。

为什么用混合检索：只靠向量，对「38 hours」「Fair Work Information Statement」、条款名这类精确词很弱；全文检索能接住这些。RRF 合并两个排名不需要分数归一化，也不需要调权重。

### 3. 回答与拒答

prompt 把检出的 5 个分块列为带编号的来源。系统提示词要求模型只根据这些来源作答、数字原样引用、按来源编号标注引用，并返回如下结构的 JSON：

```json
{"answer": "string", "citations": [1, 3], "confidence": "high | medium | low"}
```

严格的 `json_schema` 响应格式保证这个结构。temperature 为 0，这样多次评测之间可以比较。

满足以下任一条件，助手就拒答，回复「The documents I have do not cover this.」：

- `best_similarity < RAG_MIN_SIMILARITY`（默认 0.30，在黄金测试集上调出来的）；
- 模型没有返回任何有效的引用编号；
- 模型没有返回可用的 JSON。有严格 schema 约束后，只剩安全拒答或回复被截断这两种原因，重试也解决不了，所以直接拒答，并在日志里标 `error=true`。

相似度检查在调用模型之前进行，所以跑题问题不花对话 token。不在 `1..5` 范围内的引用编号会被丢弃；一个都不剩就拒答。保留下来的每条引用以文档名、页码和分块文本返回，读者可以对照原 PDF 核实。

### 4. 评测

`evals/golden.yaml` 里有 15 个分布在 11 份文档上的可回答问题，加 3 个必须拒答的跑题问题：

```yaml
- id: annual-leave-weeks
  question: How many weeks of paid annual leave does a full-time employee get each year?
  expected_doc: Annual-leave
  gold_answer: 4 weeks
- id: off-topic-visa
  question: How do I apply for a 482 visa?
  expected_doc: null          # must refuse
```

`python -m rag.eval` 把每个问题依次过一遍 `retrieve` 和 `answer`，然后报告：

| 指标 | 定义 | 目标 |
|---|---|---|
| hit@5 | `expected_doc` 在检出的 5 个分块所属文档之中 | ≥ 80% |
| cited hit | `expected_doc` 在实际被引用的文档之中 | 只报告 |
| faithfulness | 由 LLM 当评判（同一个对话模型），对照检出来源和标准答案给回答打 1–5 分 | 平均 ≥ 4 |
| refusal rate | 跑题问题中被拒答的比例 | 3 / 3 |
| latency | 每个问题的平均和 p95 毫秒数 | 只报告 |
| cost | 按标价算的每个问题平均美元费用 | 只报告 |

每次运行都把一张 Markdown 表写到 `evals/results/<UTC 时间戳>.md`。这些文件会提交进仓库，所以本文引用的数字可以复现。

`python -m rag.eval --retrieval-only` 只跑检索这一半：hit@5 和每道题的最高相似度，花费不到一美分。先用它调分块和 `RAG_MIN_SIMILARITY`，再花钱跑带评判的完整评测。

### 5. 费用与延迟核算

每次 `/ask` 都往 `queries` 表写一行：问题、是否拒答、检出的文档、被引用的分块、prompt / completion / embedding 的 token 数、美元费用和延迟。费用由 API 返回的 token 用量乘以 settings 里的价格表算出（单位：每百万 token 的美元数，注明是近似值、可覆盖）：嵌入 0.02；`gpt-4.1-mini` 输入 0.40、输出 1.60。按这个价，完整评测跑一轮实际花了 0.029 美元。

### 6. 数据模型

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

`DATABASE_URL` 默认是 `postgresql://rag:rag@127.0.0.1:5433/rag`。用 5433 端口是为了不和本机 5432 上的 PostgreSQL 撞车。地址写 `127.0.0.1` 而不是 `localhost`，是因为 Windows 上 `localhost` 会先试 IPv6，而 Docker 只在 IPv4 上监听这个端口，结果每次连接都会卡上好几分钟。

### 7. HTTP API

| 端点 | 行为 |
|---|---|
| `GET /` | 演示页：带示例问题的提问框、带「Answered / Refused」标记的回答、引用卡片，以及可折叠的 5 个检索来源表 |
| `POST /ask`，请求体 `{"question": "..."}` | 返回 `answer`、`refused`、`confidence`、`citations[]`（编号、文档、页码、片段）、`sources[]`（融合后的 top-5：编号、文档、页码、片段、相似度，只被全文检索找到的为 null、是否被引用）、`latency_ms`、`cost_usd`，并记录这次查询 |
| `GET /health` | 检查数据库连接 |

错误码：数据库挂了 → 503；模型服务重试后仍失败 → 502，附一句简短说明；请求体不合法（比如问题为空或超过 1000 字符）→ 422。模型调用遇到 429 和 5xx 时，由 OpenAI SDK 按指数退避重试三次；数据库连接 5 秒连不上就放弃。

## 完整用法

完成快速上手之后，用虚拟环境里的 Python 执行（Windows 是 `.venv\Scripts\python`，其他系统 `source .venv/bin/activate` 之后用 `python`）：

```bash
# 命令行提问（每次回答约 0.001 美元）
python -m rag.retrieve "How many hours is a full-time week?"      # 只检索，不调模型
python -m rag.answer "How many weeks of paid annual leave does a full-time employee get?"
python -m rag.answer "How do I apply for a 482 visa?"             # 会拒答

# uvicorn 运行时调用 API
curl -X POST 127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How much notice does an employer have to give?"}'

# 测试与评测
pytest                               # 不联网：测试里任何 OpenAI 调用都会被拦下
python -m rag.eval --retrieval-only  # 只算检索指标，花费不到一美分
python -m rag.eval                   # 带评判的完整评测，约 0.03 美元
```

Windows PowerShell 5.1 里的 `curl` 是 `Invoke-WebRequest` 的别名，参数不一样，调 API 要这样写：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" -Body '{"question": "How much notice does an employer have to give?"}' | ConvertTo-Json -Depth 5
```

单元测试覆盖纯函数：`chunk_page`（不超大小上限、保留重叠、优先段落断开、不丢文本）、`rrf_merge`（两个列表都出现的分块排名高于只出现一次的，并列保持输入顺序）和 `validate_citations`。拒答规则直接通过 `answer()` 测试，检索和模型都换成假的：相似度过低时不调模型直接拒答，引用为空或越界则拒答，JSON 不可用则拒答并标 `error=true`。一个测试夹具会拦下所有真实的 OpenAI 调用，所以 `pytest` 永远不花钱。`tests/test_api.py` 用 FastAPI 的测试客户端加假的流水线测 HTTP 层：演示页能返回、`/ask` 字段齐全且 `cited` 标记和引用一致、不合法的问题返回 422。

## 配置

所有配置都在 `.env` 里，该文件已在 .gitignore 中。`.env.example` 说明每一项。

| 变量 | 默认值 | 用途 |
|---|---|---|
| `OPENAI_API_KEY` | 必填 | 回答、评测评判和向量嵌入 |
| `CHAT_MODEL` | `gpt-4.1-mini` | 回答和评判用的模型 |
| `EMBED_MODEL` | `text-embedding-3-small` | 嵌入模型；它的 1536 维必须和表结构里的 `vector(1536)` 一致 |
| `DATABASE_URL` | `postgresql://rag:rag@127.0.0.1:5433/rag` | PostgreSQL 连接串 |
| `RAG_MIN_SIMILARITY` | `0.30` | 最高余弦相似度低于此值就拒答 |

## 评测结果

最近一次：[evals/results/2026-11-02T10-29-04Z.md](evals/results/2026-11-02T10-29-04Z.md)，模型 `gpt-4.1-mini` 和 `text-embedding-3-small`，`RAG_MIN_SIMILARITY` 为 0.30，共 18 道黄金测试题。

| 指标 | 目标 | 最近一次 |
|---|---|---|
| hit@5 | ≥ 80% | 15/15（100%） |
| cited hit | 只报告 | 15/15（100%） |
| faithfulness（平均，1–5） | ≥ 4 | 4.80，基于 15 个已回答的题 |
| refusal rate（跑题问题） | 3 / 3 | 3/3 |
| 可回答却被拒答的题 | 只报告 | 0 |
| latency（平均 / p95） | 只报告 | 1314 / 2678 毫秒 |
| 每题费用（按标价） | 只报告 | 0.00083 美元 |

整轮按标价花了 0.029 美元：回答 0.015，评判 0.014。两道跑题题的相似度约 0.25，被相似度阈值直接拒答，没有调用模型。第三道问 482 签证，相似度 0.34，过了阈值，但模型一条来源都没引用，所以同样被拒答。

**唯一答错的一题，以及原因。** 裁员补偿那题得了 2 分：系统说工作 5 年补 4 周工资，说明文档写的是 10 周。检索是对的，正确的分块也被引用了，错在分块里的文字。那张表每一行的年限范围都折成两行，pypdf 却先输出第二行（「At least / 6 years / 5 years but less than 10 weeks」），模型读到的是错位的行。pypdf 的版面保留模式能让表格行恢复顺序，但会把页面的左右两栏文字逐行交错，不能直接换上；能识别表格的抽取方式列在「上生产还需要补什么」里。

搭建第一阶段的总花费，按 API 返回的 token 用量计算：约 0.03 美元。

## 设计取舍

- **不用编排框架。** LangChain 或 LlamaIndex 能省几百行代码，但会把这里最要紧的几步藏起来：文本怎么切、两个排名怎么合、什么时候拒答。手写让每一步都是一个带单元测试的小函数。代价是换组件要写代码，而不是改配置。
- **用 PostgreSQL + pgvector，不用专门的向量数据库。** 一个容器同时放向量、全文索引和查询日志，混合检索一次 SQL 往返就完成，运维也是熟悉的那套。pgvector 的 HNSW 对 `vector` 类型最多 2000 维，在几千万向量的规模上也比不过专用存储，但这两点对几百个分块都无关紧要。
- **约 1800 字符、重叠 200、先按段落再按句子切分。** 大到一个事实和它的限定条件通常在同一块里，小到 5 个分块能宽松地放进 prompt。重叠避免横跨边界的句子被切丢。分块不跨页，引用因此能精确到页，代价是偶尔会切断一张表格。
- **混合检索用倒数排名融合合并（k = 60）。** 向量检索管同义改写，全文检索管精确词和数字。RRF 不需要分数归一化，也不需要调权重，在只有 18 个问题可调参的情况下这很重要。全文检索把问题里的词用 OR 连接，因为 AND 要求一整句问题的每个词都出现在同一张卡片里。固定取 top-20 和 top-5 简单，但不自适应。
- **相似度阈值加引用校验双重拒答。** 两个独立信号：检索侧（最高余弦相似度低于 0.30）和生成侧（没有有效引用）。在黄金测试集上，可回答问题的相似度都不低于 0.53，跑题问题都不高于 0.34，所以 0.30 在可回答一侧留足了余量；签证那题越过了阈值，正说明还需要引用检查这第二道关。阈值仍是整个系统里最脆弱的参数，换嵌入模型就得重调。
- **引用是来源编号，由服务端映射回去。** 模型只能输出 `1..5` 这几个数字，服务端把它们映射成真实的分块、页码和文本。模型编不出不存在的文档或页码，越界编号直接丢弃。
- **严格 JSON schema 加 temperature 0。** 结构化输出保证响应可解析、结构正确，所以不需要「解析失败再重试」的循环；temperature 0 让多次评测可比。但采样确定不等于模型确定，不同运行之间仍会有细微差异。
- **选 `gpt-4.1-mini` 而不是 `gpt-5` 系列。** `gpt-5` 系列只接受默认 temperature，会让评测更嘈杂。`gpt-4.1-mini` 便宜（每百万 token 输入 0.40、输出 1.60 美元），从 5 段短来源里作答也足够强。
- **一个服务商包办全部。** 最初的设计是 Gemini 免费档负责对话、OpenAI 负责嵌入。实测这个项目上的免费档每天只允许 20 次对话请求，而评测一轮要 33 次，于是对话也换到了 OpenAI。一把 key、一个客户端、每轮评测约 0.03 美元。
- **让 LLM 对照标准答案当评判。** 评判能看到检出的来源和预期答案，1–5 分因此有锚点。回答和评判用同一个模型会有自我偏好；检索类指标（hit@5、cited hit）不依赖评判，生产环境会换一个评判模型并加人工抽查。
- **黄金测试集很小。** 18 个问题足以发现回归、让每次运行都便宜，但在统计上不算强的质量度量。
- **用一个静态演示页，不上前端框架。** 一个 HTML 文件，CSS 和 JavaScript 内联，由 FastAPI 托管，调的是评测用的同一个 `/ask`。它把检索到的来源和引用并排显示，混合检索和拒答可以现场演示。没有历史记录、流式输出和登录；见 [docs/design/2026-11-03-demo-page-design.zh.md](docs/design/2026-11-03-demo-page-design.zh.md)。

## 上生产还需要补什么

第一阶段刻意不做的：

- 真正的前端（演示页没有历史记录、流式输出和登录）、鉴权、多租户；
- 能识别表格的 PDF 抽取，因为 pypdf 会把折成两行的表格单元格打乱（见评测结果）；
- 在融合后的 top-k 上做重排（reranker），以及嵌入和回答的缓存；
- 支持追问的对话记忆；
- 换一个更强、不同的模型当评判，并加人工抽查；
- 文档更新流水线，因为 FWO 的说明文档会变（比如最低工资每年 7 月 1 日调整）；
- 对延迟、费用、拒答率的监控和告警；
- 更大、带版本的黄金测试集。

## 路线图：第二阶段

`retrieve()` 和 `answer()` 成为一个 agent 的头两个工具，agent 放在新的 `agent/` 包里，带工具调用、人工介入审批和审计日志。第一阶段的模块不变。

## 仓库结构

```
au-workplace-assistant/
  README.md  README_CN.md
  .env.example  .gitignore  requirements.txt  pyproject.toml
  docker-compose.yml          # pgvector/pgvector:pg16，端口 5433
  api.py
  static/index.html           # 演示页，由 GET / 托管
  rag/  settings.py providers.py chunking.py db.py ingest.py retrieve.py answer.py eval.py
  scripts/fetch_corpus.py
  evals/golden.yaml  evals/results/
  tests/  test_chunking.py test_retrieve.py test_answer.py test_api.py
  data/raw/                   # 下载的 PDF，已 git-ignore
  docs/design/                # 设计稿（中英文）
  docs/tasks/                 # 实施计划
```

## 语料、版权与免责声明

语料是 11 份 FWO 说明文档，构建时从 `https://www.fairwork.gov.au/sites/default/files/migration/` 下载：

| 文档 | 路径 |
|---|---|
| Introduction to the National Employment Standards（国家就业标准简介） | `723/Introduction-to-the-national-employment-standards.pdf` |
| Maximum weekly hours（每周最长工时） | `723/Maximum-weekly-hours.pdf` |
| Annual leave（年假） | `723/Annual-leave.pdf` |
| Public holidays（公共假期） | `723/Public-holidays.pdf` |
| Notice of termination and redundancy pay（解雇通知与裁员补偿） | `723/Notice-of-termination-and-redundancy-pay.pdf` |
| Minimum wages（最低工资） | `723/Minimum-wages.pdf` |
| Long service leave（长期服务假） | `723/Long-service-leave.pdf` |
| Community service leave（社区服务假） | `723/Community-service-leave.pdf` |
| Unpaid work（无薪工作） | `723/Unpaid-work.pdf` |
| Record-keeping and pay slips（记录保存与工资单） | `723/Record-keeping-and-pay-slips.pdf` |
| Fair Work Information Statement（公平工作信息声明） | `724/Fair-Work-Information-Statement.pdf` |

原始材料版权归澳大利亚联邦（Fair Work Ombudsman）所有。仓库只在本地保存派生出来的分块，不再分发 PDF。另外放进 `data/raw/` 的任何 `.pdf` 也会以同样方式入库。

本项目是技术演示，不构成法律建议。权益标准会随时间变化，请以 [fairwork.gov.au](https://www.fairwork.gov.au) 的最新数字为准。OpenAI API 的数据默认不用于模型训练。

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。
