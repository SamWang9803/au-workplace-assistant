"""Golden-set evaluation: hit@5, cited hit, faithfulness, refusal rate, latency and cost."""
import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml

from rag.answer import Answer, answer, format_sources
from rag.providers import chat_json, cost_usd
from rag.retrieve import retrieve
from rag.settings import settings

GOLDEN = Path("evals/golden.yaml")
RESULTS = Path("evals/results")

JUDGE_PROMPT = """You grade a question-answering system that must answer only from Australian workplace fact sheets.
You get the question, a reference answer, the system's answer and the numbered sources the system retrieved.
Score the system's answer from 1 to 5:
5 = every claim is supported by the sources and it agrees with the reference answer;
4 = supported and correct, with a minor omission or harmless extra detail;
3 = partly correct, or includes a claim the sources do not support;
2 = mostly wrong or mostly unsupported;
1 = contradicts the sources or the reference answer.
Give the reason in one sentence."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer", "enum": [1, 2, 3, 4, 5]}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
    "additionalProperties": False,
}


def judge(question: str, gold: str, result: Answer) -> tuple[int | None, str, float]:
    """Score one answer against the retrieved sources and the reference answer; return score, reason and cost."""
    user = (
        f"Question: {question}\n\nReference answer: {gold}\n\nSystem answer: {result.text}\n\n"
        f"Sources the system retrieved:\n\n{format_sources(result.sources)}"
    )
    reply = chat_json(JUDGE_PROMPT, user, JUDGE_SCHEMA)
    cost = cost_usd(reply.prompt_tokens, reply.completion_tokens, 0)
    if reply.data is None:
        return None, "the judge returned no usable JSON", cost
    return reply.data["score"], reply.data["reason"], cost


def evaluate(items: list[dict]) -> tuple[list[dict], float]:
    """Answer every golden question, judge the answered ones, and return per-question rows and the judging cost."""
    rows, judge_cost = [], 0.0
    for item in items:
        expected = item.get("expected_doc")
        result = answer(item["question"])
        row = {
            "id": item["id"],
            "expected": expected or "(off-topic)",
            "similarity": result.best_similarity,
            "hit": expected in {c.document for c in result.sources} if expected else None,
            "cited": expected in {c.document for c in result.citations} if expected else None,
            "refused": result.refused,
            "score": None,
            "reason": "",
            "latency_ms": result.latency_ms,
            "cost": result.cost_usd,
        }
        if expected and not result.refused:
            row["score"], row["reason"], cost = judge(item["question"], item["gold_answer"], result)
            judge_cost += cost
        rows.append(row)
        print(f"{item['id']:<34} refused={str(result.refused):<5} faithfulness={row['score'] or '-'}  {result.latency_ms} ms")
    return rows, judge_cost


def _ratio(n: int, total: int) -> str:
    """Format n out of total with a percentage."""
    return f"{n}/{total} ({n / total:.0%})" if total else "n/a"


def _mark(value: bool | None) -> str:
    """Render True, False and None as yes, no and a dash."""
    return "-" if value is None else ("yes" if value else "no")


def p95(values: list[int]) -> int:
    """Return the 95th percentile by the nearest-rank method."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def summarize(rows: list[dict]) -> list[tuple[str, str, str]]:
    """Compute the headline metrics as (metric, result, target) rows."""
    answerable = [r for r in rows if r["hit"] is not None]
    off_topic = [r for r in rows if r["hit"] is None]
    scores = [r["score"] for r in answerable if r["score"] is not None]
    latencies = [r["latency_ms"] for r in rows]
    faithfulness = f"{sum(scores) / len(scores):.2f} over {len(scores)} answered" if scores else "n/a"
    return [
        ("hit@5", _ratio(sum(r["hit"] for r in answerable), len(answerable)), ">= 80%"),
        ("cited hit", _ratio(sum(r["cited"] for r in answerable), len(answerable)), "reported"),
        ("faithfulness (mean, 1-5)", faithfulness, "mean >= 4"),
        ("refusal rate (off-topic)", _ratio(sum(r["refused"] for r in off_topic), len(off_topic)), "3 of 3"),
        ("answerable questions refused", str(sum(r["refused"] for r in answerable)), "reported"),
        ("latency (mean / p95)", f"{sum(latencies) / len(latencies):.0f} / {p95(latencies)} ms", "reported"),
        ("cost per question (list price)", f"${sum(r['cost'] for r in rows) / len(rows):.5f}", "reported"),
    ]


def write_results(rows: list[dict], summary: list[tuple[str, str, str]], judge_cost: float) -> Path:
    """Write the summary and per-question tables to evals/results/<UTC timestamp>.md and return the path."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    lines = [
        f"# Evaluation run {stamp}",
        "",
        f"Chat model `{settings.chat_model}`, embedding model `{settings.embed_model}`, "
        f"RAG_MIN_SIMILARITY {settings.rag_min_similarity}, {len(rows)} questions.",
        "",
        "| Metric | Result | Target |",
        "|---|---|---|",
        *(f"| {metric} | {result} | {target} |" for metric, result, target in summary),
        "",
        f"Run cost at list price: ${sum(r['cost'] for r in rows):.4f} for answering plus ${judge_cost:.4f} for judging.",
        "",
        "## Per question",
        "",
        "| id | expected document | best similarity | hit@5 | cited | refused | faithfulness | latency ms | judge's reason |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        score = "-" if r["score"] is None else r["score"]
        lines.append(
            f"| {r['id']} | {r['expected']} | {r['similarity']:.3f} | {_mark(r['hit'])} | {_mark(r['cited'])} | "
            f"{_mark(r['refused'])} | {score} | {r['latency_ms']} | {r['reason'].replace('|', '/')} |"
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{stamp}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def retrieval_only(items: list[dict]) -> None:
    """Retrieve for every question without the chat model; print hit@5 and similarities for tuning the threshold."""
    hits, answerable, off_topic = 0, [], []
    for item in items:
        found = retrieve(item["question"])
        expected = item.get("expected_doc")
        hit = expected in {c.document for c in found.chunks} if expected else None
        hits += bool(hit)
        (answerable if expected else off_topic).append(found.best_similarity)
        print(f"{item['id']:<34} best similarity {found.best_similarity:.3f}  hit@5 {_mark(hit)}")
    print(f"\nhit@5 {_ratio(hits, len(answerable))}")
    print(f"lowest similarity among answerable questions: {min(answerable):.3f}")
    if off_topic:
        print(f"highest similarity among off-topic questions: {max(off_topic):.3f}")
    print(f"RAG_MIN_SIMILARITY is {settings.rag_min_similarity}")


def main() -> None:
    """Run the full evaluation, or the near-free retrieval-only pass with --retrieval-only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-only", action="store_true", help="skip the chat model; costs a fraction of a cent")
    args = parser.parse_args()
    items = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    if args.retrieval_only:
        retrieval_only(items)
        return
    rows, judge_cost = evaluate(items)
    summary = summarize(rows)
    path = write_results(rows, summary, judge_cost)
    for metric, result, target in summary:
        print(f"{metric:<32} {result:<28} target {target}")
    print(f"results written to {path}")


if __name__ == "__main__":
    main()
