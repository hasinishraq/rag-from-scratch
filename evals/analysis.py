"""Detailed breakdown of one evaluation run, for the Evaluation page.

Pure functions over a run file (written by scripts/run_eval.py) and its question set, so the
numbers can be tested without the app. Rates come with a standard error, because with a few
hundred questions a difference of a few points is not conclusive (ENG-RAG-001 §8.3).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from evals.metrics import Question, QuestionResult, changed_questions, reciprocal_rank

MAX_MISSES = 60
MAX_CHANGES = 40
LATENCY_BUCKETS = 12


def std_error(rate: float | None, n: int) -> float | None:
    if rate is None or n == 0:
        return None
    return round(math.sqrt(max(rate * (1 - rate), 0.0) / n), 4)


def gold_rank(retrieved: Sequence[str], gold: Sequence[str]) -> int | None:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return rank
    return None


def _percentile(sorted_values: Sequence[int], p: float) -> int | None:
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, int(p * len(sorted_values)))]


def latency_summary(values: Sequence[int]) -> dict[str, Any]:
    """Percentiles plus a histogram scaled to the 99th percentile; slower outliers go into a
    final bucket, so one slow question cannot squash the rest of the chart."""
    v = sorted(values)
    if not v:
        return {"n": 0, "hist": []}
    lo, cap = v[0], _percentile(v, 0.99) or v[-1]
    width = max(1, math.ceil((cap - lo + 1) / LATENCY_BUCKETS))
    counts = Counter(min(LATENCY_BUCKETS, (x - lo) // width) for x in v)
    hist = [
        {"from_ms": lo + i * width, "to_ms": lo + (i + 1) * width, "count": counts.get(i, 0)}
        for i in range(LATENCY_BUCKETS)
    ]
    if counts.get(LATENCY_BUCKETS):
        hist.append(
            {
                "from_ms": lo + LATENCY_BUCKETS * width,
                "to_ms": v[-1],
                "count": counts[LATENCY_BUCKETS],
                "overflow": True,
            }
        )
    return {
        "n": len(v),
        "min": lo,
        "max": v[-1],
        "mean": round(statistics.fmean(v)),
        "p50": _percentile(v, 0.5),
        "p90": _percentile(v, 0.9),
        "p95": _percentile(v, 0.95),
        "p99": _percentile(v, 0.99),
        "hist": hist,
    }


def _doc_of(chunk_id: str | None) -> str | None:
    return chunk_id.rsplit("_", 1)[0] if chunk_id and "_" in chunk_id else None


def _retrieval(
    results: Sequence[QuestionResult], qs: dict[str, Question]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-question rank of the first gold chunk, and the curves built from it."""
    rows: list[dict[str, Any]] = []
    for r in results:
        q = qs.get(r.id)
        if q is None or not r.answerable or not q.gold_chunk_ids:
            continue
        rows.append({"result": r, "question": q, "rank": gold_rank(r.retrieved, q.gold_chunk_ids)})
    depth = max((len(x["result"].retrieved) for x in rows), default=0)
    n = len(rows)
    recall_at_k = (
        [
            {"k": k, "value": round(sum(1 for x in rows if x["rank"] and x["rank"] <= k) / n, 4)}
            for k in range(1, depth + 1)
        ]
        if n
        else []
    )
    ranks = Counter(x["rank"] for x in rows)
    hist = [{"rank": str(k), "count": ranks.get(k, 0)} for k in range(1, depth + 1)]
    hist.append({"rank": "not found", "count": ranks.get(None, 0)})
    for h in hist:
        h["share"] = round(h["count"] / n, 4) if n else 0.0
    return rows, {"n_scored": n, "depth": depth, "recall_at_k": recall_at_k, "rank_hist": hist}


def _by_topic(
    rows: list[dict[str, Any]], results: Sequence[QuestionResult], qs: dict[str, Question]
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in rows:
        groups[str(getattr(x["question"], "topic", None) or "All questions")].append(x)
    unanswerable: Counter[str] = Counter(
        str(getattr(qs[r.id], "topic", None) or "All questions")
        for r in results
        if not r.answerable and r.id in qs
    )
    out = []
    for topic, xs in sorted(groups.items()):
        hits = sum(1 for x in xs if x["rank"] and x["rank"] <= 5)
        out.append(
            {
                "topic": topic,
                "n": len(xs),
                "unanswerable": unanswerable.get(topic, 0),
                "recall_at_5": round(hits / len(xs), 4),
                "mrr": round(
                    statistics.fmean(
                        reciprocal_rank(x["result"].retrieved, x["question"].gold_chunk_ids)
                        for x in xs
                    ),
                    4,
                ),
                "misses": len(xs) - hits,
            }
        )
    return out


def _generation(results: Sequence[QuestionResult], retrieval_only: bool) -> dict[str, Any] | None:
    if retrieval_only:
        return None
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    faithful = [r.faithful for r in results if r.faithful is not None and not r.refused]
    correct = [r.correct for r in results if r.correct is not None]
    return {
        "answered": sum(1 for r in results if not r.refused and not r.error),
        "refused": sum(1 for r in results if r.refused),
        "errors": sum(1 for r in results if r.error),
        "correct_refusals": sum(1 for r in unanswerable if r.refused),
        "unanswerable": len(unanswerable),
        "false_refusals": sum(1 for r in answerable if r.refused),
        "answerable": len(answerable),
        "judged_faithful": len(faithful),
        "faithful": sum(1 for f in faithful if f),
        "judged_correct": len(correct),
        "correct": sum(1 for c in correct if c),
        "providers": dict(Counter(r.provider or "none" for r in results)),
        "tokens": sum(r.tokens for r in results),
    }


def analyse(
    run: dict[str, Any], qs: dict[str, Question], previous: dict[str, Any] | None = None
) -> dict[str, Any]:
    results = [QuestionResult.model_validate(r) for r in run["results"]]
    rows, retrieval = _retrieval(results, qs)
    metrics = run["metrics"]
    n_ans = retrieval["n_scored"]
    n_un = sum(1 for r in results if not r.answerable)
    stderr = {
        "recall_at_5": std_error(metrics.get("recall_at_5"), n_ans),
        "mrr": None,  # not a proportion; its spread is shown by the rank histogram
        "faithfulness": std_error(metrics.get("faithfulness"), n_ans),
        "answer_correctness": std_error(metrics.get("answer_correctness"), n_ans),
        "refusal_accuracy": std_error(metrics.get("refusal_accuracy"), n_un),
        "false_refusal_rate": std_error(metrics.get("false_refusal_rate"), n_ans),
    }
    misses = [
        {
            "id": x["result"].id,
            "question": x["question"].question,
            "topic": getattr(x["question"], "topic", None),
            "rank": x["rank"],
            "gold": x["question"].gold_chunk_ids,
            "first": x["result"].retrieved[0] if x["result"].retrieved else None,
        }
        for x in rows
        if not x["rank"] or x["rank"] > 5
    ]
    for m in misses:
        # A neighbouring section of the right document ranked first is a ranking problem;
        # another document ranked first is a retrieval problem.
        m["same_document"] = _doc_of(m["first"]) in {_doc_of(g) for g in m["gold"]}
    same_doc = sum(1 for m in misses if m["same_document"])
    misses = misses[:MAX_MISSES]
    out: dict[str, Any] = {
        "run": run["run"],
        "config": run.get("config"),
        "split": run.get("split"),
        "retrieval_only": bool(run.get("retrieval_only")),
        "questions_file": run.get("questions"),
        "n": len(results),
        "n_answerable": n_ans,
        "n_unanswerable": n_un,
        "matched_questions": sum(1 for r in results if r.id in qs),
        "metrics": metrics,
        "stderr": stderr,
        **retrieval,
        "by_topic": _by_topic(rows, results, qs),
        "latency": latency_summary([r.latency_ms for r in results if r.error is None]),
        "generation": _generation(results, bool(run.get("retrieval_only"))),
        "misses": misses,
        "misses_same_document": same_doc,
        "previous": None,
        "changes": [],
    }
    if previous is not None:
        prev = [QuestionResult.model_validate(r) for r in previous["results"]]
        gold = {q.id: q.gold_chunk_ids for q in qs.values()}
        out["previous"] = {
            "run": previous["run"],
            "config": previous.get("config"),
            "metrics": previous["metrics"],
        }
        out["changes"] = changed_questions(prev, results, gold)[:MAX_CHANGES]
    return out
