"""Run the labelled evaluation set through pipeline.answer() and report the metrics table.

Usage: python scripts/run_eval.py --config configs/default.yaml [--retrieval-only] [--judge]
       [--split dev|test|all] [--compare evals/results/<previous>.json] [--validate]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.judge import judge
from evals.metrics import (
    Question,
    QuestionResult,
    changed_questions,
    load_questions,
    summarise,
)

from rag.config import load_config
from rag.errors import RagError
from rag.pipeline import Pipeline
from rag.services import build_services
from rag.types import Mode


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--questions", default="evals/questions.jsonl")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--retrieval-only", action="store_true", help="no generation, no quota")
    ap.add_argument("--judge", action="store_true", help="grade answers with an LLM judge")
    ap.add_argument("--judge-provider", default=None, help="provider name for the judge")
    ap.add_argument("--mode", default=None, choices=[m.value for m in Mode])
    ap.add_argument("--compare", default=None, help="previous results file")
    ap.add_argument("--validate", action="store_true", help="check the file and gold ids only")
    ap.add_argument("--out", default="evals/results")
    return ap.parse_args()


def _validate(questions: list[Question], pipeline: Pipeline) -> int:
    known = pipeline.s.chunks.get_many({c for q in questions for c in q.gold_chunk_ids})
    problems = [
        f"{q.id}: unknown gold chunk {c}"
        for q in questions
        for c in q.gold_chunk_ids
        if c not in known
    ]
    problems += [
        f"{q.id}: answerable but no gold_chunk_ids"
        for q in questions
        if q.answerable and not q.gold_chunk_ids
    ]
    ids = [q.id for q in questions]
    problems += [f"duplicate id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for p in problems:
        print(p)
    print(f"{len(questions)} questions, {len(problems)} problems")
    return 1 if problems else 0


def _run_one(pipeline: Pipeline, q: Question, a: argparse.Namespace) -> QuestionResult:
    mode = Mode(a.mode) if a.mode else None
    try:
        ans = pipeline.answer(q.question, mode=mode, retrieve_only=a.retrieval_only)
    except RagError as exc:
        return QuestionResult(
            id=q.id,
            answerable=q.answerable,
            retrieved=[],
            refused=False,
            answer="",
            latency_ms=0,
            error=f"{exc.code}: {exc.message}",
        )
    faithful = correct = None
    if a.judge and not a.retrieval_only and not ans.refused:
        try:
            faithful, correct = judge(
                pipeline.s.gateway,
                q.question,
                ans.answer,
                q.reference_answer,
                ans.sources,
                a.judge_provider,
                mode,
            )
        except RagError as exc:
            logging.warning("judge failed on %s: %s", q.id, exc.message)
    trace = pipeline.s.chunks.get_trace(ans.trace_id)
    tokens = (trace.tokens_in + trace.tokens_out) if trace and not trace.cached else 0
    return QuestionResult(
        id=q.id,
        answerable=q.answerable,
        retrieved=[s.chunk_id for s in ans.sources],
        refused=ans.refused,
        answer=ans.answer,
        latency_ms=ans.latency_ms,
        provider=ans.provider,
        tokens=tokens,
        faithful=faithful,
        correct=correct,
    )


def main() -> int:
    # Windows consoles default to cp1252; model output often contains other characters.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    a = _args()
    logging.basicConfig(level=logging.WARNING)
    qpath = Path(a.questions)
    if not qpath.exists():
        print(
            f"{qpath} not found. Write the evaluation set first (task T4.1); see "
            "evals/questions.example.jsonl for the format."
        )
        return 2
    questions = load_questions(qpath)
    cfg = load_config(a.config)
    pipeline = Pipeline(build_services(cfg))
    if a.validate:
        return _validate(questions, pipeline)
    chosen = [q for q in questions if a.split == "all" or q.split == a.split]
    results = []
    for i, q in enumerate(chosen, start=1):
        results.append(_run_one(pipeline, q, a))
        print(f"\r{i}/{len(chosen)}", end="", flush=True)
    print()
    metrics = summarise(chosen, results)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = {
        "run": stamp,
        "config": a.config,
        "config_hash": cfg.hash(),
        "split": a.split,
        "retrieval_only": a.retrieval_only,
        "questions": Path(a.questions).as_posix(),
        "metrics": metrics,
        "results": [r.model_dump() for r in results],
    }
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{stamp}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{'metric':<22}value")
    for k, v in metrics.items():
        print(f"{k:<22}{v}")
    if a.compare:
        old = json.loads(Path(a.compare).read_text(encoding="utf-8"))
        prev = [QuestionResult.model_validate(r) for r in old["results"]]
        gold = {q.id: q.gold_chunk_ids for q in chosen}
        diff = changed_questions(prev, results, gold)
        print(f"\nchanged questions vs {a.compare}: {len(diff)}")
        for line in diff:
            print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
