# 设计：Fair Work RAG 助手（第一阶段）

日期：2026-11-02 · 状态：已在对话中拍板，待实现

修订 2026-11-02：embedding 从 OpenAI 的 `text-embedding-3-small` 改为 Gemini 的
`gemini-embedding-2`；整个项目只用一个服务商、一把 key。

英文原版：`2026-11-02-fairwork-rag-design.md`（两份内容有出入时以英文版为准）

## 1. 目标

基于 Fair Work Ombudsman（FWO，澳大利亚公平工作监察署）的说明文档（fact sheet），
回答关于澳大利亚最低职场权益的问题。每个回答都要注明出处文档和页码；文档没覆盖到的
问题要拒答；并附一张评测表，说明它答得有多好。这是一个个人作品集项目：每个组件都
必须能在面试里讲清楚。

第二阶段（之后再做，不在本文范围内）会在同一套检索和回答模块之上，加一个带工具和
人工介入（human-in-the-loop）的 agent。

### 第一阶段不做的事

不做 Web 界面、不做鉴权、不做多租户、不做重排（reranker）、不做缓存、不做对话记忆。
这些会在 README 里列为「上生产还需要补的东西」。

## 2. 语料

十一份 FWO 的 PDF，由 `scripts/fetch_corpus.py` 在构建时下载到 `data/raw/`
（已加入 .gitignore，不随仓库分发）。2026-11-02 已逐一确认可在
`https://www.fairwork.gov.au/sites/default/files/migration/` 下访问：

| 文件 | 路径 |
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

另外放进 `data/raw/` 的任何 `.pdf` 也按同样方式入库。原始材料版权归澳大利亚联邦
（Fair Work Ombudsman）所有；仓库只在本地存派生出来的分块（chunk），并在 README 里
注明来源。

## 3. 架构

一个 Python 3.12 进程，加一个带 pgvector 的 PostgreSQL 16 容器。不用编排框架：
流水线手写，这样每一步都看得见、测得到。

```
PDF ──入库(ingest)──▶ chunks（text、embedding、tsvector）存入 PostgreSQL
问题 ──embed──▶ 向量检索 top-20 ─┐
问题 ──websearch_to_tsquery──▶ 全文检索 top-20 ─┴─ RRF ──▶ top-5 chunks
top-5 + 问题 ──prompt──▶ Gemini（JSON）──校验引用 / 拒答──▶ 回答
每次 /ask ──▶ queries 表（token 数、费用、延迟）
```

所有模型调用都走 `openai` SDK，指向 Gemini 的 OpenAI 兼容端点
`https://generativelanguage.googleapis.com/v1beta/openai/`（环境变量 `GEMINI_BASE_URL`），
只用一把 key `GEMINI_API_KEY`：

| 用途 | 模型（默认值，可用环境变量覆盖） |
|---|---|
| 对话与评判（judge） | `gemini-2.5-flash`，temperature 0，`response_format` 为 json_object |
| 向量嵌入 | `gemini-embedding-2`，传 `dimensions=1536`：模型按 Matryoshka 方式训练，服务端把 3072 维输出截到 1536 维并归一化后返回。用 1536 是为了保住 `vector(1536)` 列和它的 HNSW 索引（pgvector 对 `vector` 类型最多索引 2000 维）。2026-11-02 已对真实端点实测通过 |

两个模型在 Gemini 免费档都不收费。对话大约每分钟只允许 5 到 10 次请求，所以对话
客户端强制两次调用之间留最小间隔（默认 7 秒，环境变量 `CHAT_MIN_INTERVAL_S`）；
对话和 embedding 调用遇到 429/5xx 都按指数退避重试三次。embedding 的限额按项目计，
在 AI Studio 里查看；每批 32 个分块，远低于外界报出的每分钟 3 万 token 免费上限。
免费档的输入 Google 可能用于改进产品；语料是公开文档，README 里会写明。

## 4. 组件

一个模块只干一件事。能写成纯函数的尽量写成纯函数，这样测试不需要联网。

| 模块 | 职责 | 接口 |
|---|---|---|
| `rag/settings.py` | 读 `.env`：`GEMINI_API_KEY`、`GEMINI_BASE_URL`、模型名、`DATABASE_URL`、分块大小、`RAG_MIN_SIMILARITY`、价格表 | `settings` 对象 |
| `rag/providers.py` | 一个指向 Gemini 端点的 `openai` 客户端，带限流和重试。`embed` 传 `dimensions=1536`。对话返回真实 token 用量；端点不返回 embedding 的用量，所以 `embed` 按字符数 / 4 估算 | `embed(texts) -> list[list[float]], usage` · `chat_json(system, user) -> dict, usage, latency_ms` |
| `rag/chunking.py` | 把一页文本切成约 1800 字符、重叠 200 字符的块，尽量在段落或句子边界断开 | `chunk_page(text) -> list[str]`（纯函数） |
| `rag/db.py` | 连接、建表、写入，以及两种检索查询 | `init_schema()`、`upsert_document()`、`insert_chunks()`、`vector_search()`、`text_search()`、`log_query()` |
| `rag/ingest.py` | 用 pypdf 逐页读 PDF，分块，按每批 32 条做嵌入，存库。读不了的文件打警告跳过。重跑会替换该文档的全部分块 | `python -m rag.ingest [data/raw]` |
| `rag/retrieve.py` | 把问题做嵌入，跑两种检索，用 reciprocal rank fusion（k = 60）合并，返回 top-5 分块和最高余弦相似度 | `retrieve(question) -> Retrieval(chunks, best_similarity)` · `rrf_merge(lists) -> ranked`（纯函数） |
| `rag/answer.py` | 拼带编号来源的 prompt，调对话模型，校验 JSON，套用拒答规则 | `answer(question) -> Answer(text, citations, refused, usage, cost_usd, latency_ms)` · `validate_citations()`（纯函数） |
| `rag/eval.py` | 跑黄金测试集，算指标，写结果表 | `python -m rag.eval` |
| `api.py` | FastAPI 应用：`POST /ask`、`GET /health` | HTTP |
| `scripts/fetch_corpus.py` | 下载十一份 PDF，报告失败项，绝不覆盖已有文件 | 命令行 |

## 5. 数据模型

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id          serial PRIMARY KEY,
  name        text UNIQUE NOT NULL,      -- 文件名，不含扩展名
  source_url  text,
  pages       int NOT NULL,
  ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
  id          serial PRIMARY KEY,
  document_id int NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page        int NOT NULL,              -- PDF 页码，从 1 起
  idx         int NOT NULL,              -- 分块在该页内的顺序
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

`DATABASE_URL` 默认是 `postgresql://rag:rag@localhost:5433/rag`；用 5433 端口，
是为了不和本机 5432 上可能已有的 PostgreSQL 撞车。

## 6. 检索

1. 把问题做嵌入（一次 Gemini embedding 调用，`dimensions=1536`）。
2. 向量检索：`ORDER BY embedding <=> $q LIMIT 20`，余弦相似度按 `1 - distance`
   返回。
3. 全文检索：`WHERE tsv @@ websearch_to_tsquery('english', $q)`，按 `ts_rank_cd`
   排序，取 20 条。
4. Reciprocal rank fusion：对两个列表求 `score(chunk) = Σ 1 / (60 + rank_in_list)`，
   取前 5。
5. `best_similarity` 是向量检索结果里的最高余弦相似度，供拒答规则使用。结果为空
   （还没入库任何东西）时按相似度 0 处理，这样问题会被拒答，而不是凭空作答。

为什么用混合检索：纯向量检索对「38 hours」「Fair Work Information Statement」、
条款名这类精确词很弱；全文检索能接住这些，而 RRF 不需要调权重。

## 7. 回答与拒答

系统提示词：只根据带编号的来源作答，数字原样引用，按编号标注引用，返回 JSON
`{"answer": str, "citations": [int], "confidence": "high"|"medium"|"low"}`。
来源里没有答案时，返回空的 `citations` 列表并明说。

满足以下任一条件就拒答（回复「The documents I have do not cover this.」）：

- `best_similarity < RAG_MIN_SIMILARITY`（默认 0.30；在黄金测试集上调出来的）；
- 模型没有返回任何有效的引用编号；
- JSON 解析失败，且用更严格的指令重试一次后仍失败（这条响应同时在日志里标
  `error=true`）。

不在 `1..5` 范围内的引用编号会被丢弃；一个都不剩就拒答。保留下来的每条引用，
以文档名、页码和分块文本的形式返回。

## 8. API

`POST /ask` 请求体 `{"question": "..."}` →

```json
{
  "answer": "...",
  "refused": false,
  "citations": [{"document": "Annual-leave", "page": 1, "snippet": "..."}],
  "latency_ms": 2140,
  "cost_usd": 0.000312
}
```

`GET /health` 检查数据库连接。每次 `/ask` 都记进 `queries` 表。错误码：数据库挂了
→ 503；模型服务重试后仍失败 → 502，附一句简短说明；请求校验不过 → 422。

## 9. 评测

`evals/golden.yaml`：

```yaml
- id: annual-leave-weeks
  question: How many weeks of paid annual leave does a full-time employee get each year?
  expected_doc: Annual-leave
  gold_answer: 4 weeks
- id: off-topic-visa
  question: How do I apply for a 482 visa?
  expected_doc: null          # 必须拒答
```

十五个可回答的问题，加三个跑题的问题。`python -m rag.eval` 把每个问题依次过一遍
`retrieve` 和 `answer`，然后报告：

| 指标 | 定义 | 目标 |
|---|---|---|
| hit@5 | `expected_doc` 在检出的五个分块所属文档之中 | ≥ 80% |
| cited hit | `expected_doc` 在被引用分块所属文档之中 | 只报告 |
| faithfulness | 由 Gemini 当评判（judge），对照检出来源和标准答案给回答打 1–5 分 | 平均 ≥ 4 |
| refusal rate | 跑题问题中被拒答的比例 | 3 / 3 |
| latency | 每个问题的平均和 p95 毫秒数 | 只报告 |
| cost | 按标价算的每个问题平均美元费用 | 只报告 |

结果以表格形式写到 `evals/results/<UTC 时间戳>.md`；README 引用最近一次的结果。
评判用的 prompt 放在 `rag/eval.py` 里，评判模型走同一个对话客户端（因此也受同样
的限流）。

## 10. 费用核算

价格表放在 settings 里，单位是每百万 token 的美元数，注明是近似值、可覆盖：
Gemini Embedding 2 为 0.20；Gemini 2.5 Flash 输入 0.30、输出 2.50。embedding 的
token 数按字符数 / 4 估算，因为端点不返回 embedding 的用量。免费档不收费，所以
README 同时写明按标价算的数字和实际花费为零。

## 11. 错误处理

- 模型服务 429/5xx（对话和 embedding）：重试三次，退避 2 秒、4 秒、8 秒；之后调用失败，由调用方报告
  （评测标记该行，API 返回 502）。
- PDF 读不了或页面为空：打警告跳过，继续。
- 模型返回的 JSON：解析、校验结构、重试一次，仍不行就拒答并标 `error=true`。
- 密钥只放在 `.env`（已 git-ignore）；`.env.example` 说明需要哪些键。

## 12. 测试

用 `pytest` 测纯函数，不联网、不连库：

- `chunk_page`：不超出大小上限，保留重叠，优先在段落处断开，绝不丢文本。
- `rrf_merge`：两个列表里都出现的分块，排名高于只在一个列表里出现的；并列时保持
  输入顺序。
- `validate_citations` 和拒答规则：越界编号被丢弃，引用为空则拒答，相似度过低则
  拒答。

第一阶段的验收标准：十一份文档都能入库并报出分块数；黄金测试集达到上面的目标；
`/ask` 返回精确到页的引用。

## 13. 仓库结构

```
au-workplace-assistant/
  README.md
  .env.example  .gitignore  requirements.txt  pyproject.toml
  docker-compose.yml          # pgvector/pgvector:pg16，端口 5433
  api.py
  rag/  settings.py providers.py chunking.py db.py ingest.py retrieve.py answer.py eval.py
  scripts/fetch_corpus.py
  evals/golden.yaml  evals/results/
  tests/  test_chunking.py test_retrieve.py test_answer.py
  data/raw/                   # 下载的 PDF，已 git-ignore
  docs/design/
```

配置项（`.env`）：`GEMINI_API_KEY`、`GEMINI_BASE_URL`、`CHAT_MODEL`、`EMBED_MODEL`、
`DATABASE_URL`、`RAG_MIN_SIMILARITY`、`CHAT_MIN_INTERVAL_S`。

## 14. 第二阶段的接口

`retrieve()` 和 `answer()` 就是之后那个 agent 的头两个工具。agent 放在新的 `agent/`
包里，不改动上面这些模块。
