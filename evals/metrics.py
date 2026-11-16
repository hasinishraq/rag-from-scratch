"""Evaluation metrics (ENG-RAG-001 §8.3). Pure functions over per-question results."""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

RECALL_K = 5


class Question(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    question: str
    gold_chunk_ids: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    answerable: bool = True
    type: str = "factual"
    split: str = "dev"


class QuestionResult(BaseModel):
    id: str
    answerable: bool
    retrieved: list[str]
    refused: bool
    answer: str
    latency_ms: int
    provider: str | None = None
    tokens: int = 0
    faithful: bool | None = None
    correct: bool | None = None
    error: str | None = None


def load_questions(path: Path) -> list[Question]:
    out: list[Question] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                out.append(Question.model_validate(json.loads(line)))
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: {exc}") from exc
    return out


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int = RECALL_K) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def summarise(
    questions: Sequence[Question], results: Sequence[QuestionResult]
) -> dict[str, float | int | None]:
    by_id = {q.id: q for q in questions}
    answerable = [r for r in results if r.answerable and by_id[r.id].gold_chunk_ids]
    unanswerable = [r for r in results if not r.answerable]
    judged_f = [r.faithful for r in results if r.faithful is not None and not r.refused]
    judged_c = [r.correct for r in results if r.correct is not None and r.answerable]
    latencies = sorted(r.latency_ms for r in results if r.error is None)
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None
    return {
        "n": len(results),
        "recall_at_5": _mean(
            [recall_at_k(r.retrieved, by_id[r.id].gold_chunk_ids) for r in answerable]
        ),
        "mrr": _mean(
            [reciprocal_rank(r.retrieved, by_id[r.id].gold_chunk_ids) for r in answerable]
        ),
        "faithfulness": _mean([float(x) for x in judged_f]),
        "answer_correctness": _mean([float(x) for x in judged_c]),
        "refusal_accuracy": _mean([float(r.refused) for r in unanswerable]),
        "false_refusal_rate": _mean([float(r.refused) for r in results if r.answerable]),
        "latency_median_ms": int(statistics.median(latencies)) if latencies else None,
        "latency_p95_ms": p95,
        "tokens_per_query": _mean([float(r.tokens) for r in results]),
        "errors": sum(1 for r in results if r.error),
    }


def changed_questions(
    old: Sequence[QuestionResult], new: Sequence[QuestionResult], gold: dict[str, list[str]]
) -> list[str]:
    """Questions whose top-5 hit, refusal or judged outcome differs between two runs."""
    before = {r.id: r for r in old}
    out: list[str] = []
    for r in new:
        prev = before.get(r.id)
        if prev is None:
            continue
        g = gold.get(r.id, [])
        hit_now, hit_then = recall_at_k(r.retrieved, g), recall_at_k(prev.retrieved, g)
        if (hit_now, r.refused, r.faithful, r.correct) != (
            hit_then,
            prev.refused,
            prev.faithful,
            prev.correct,
        ):
            out.append(
                f"{r.id}: recall {hit_then:.2f}->{hit_now:.2f}, refused "
                f"{prev.refused}->{r.refused}, faithful {prev.faithful}->{r.faithful}"
            )
    return out
