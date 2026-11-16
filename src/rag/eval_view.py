"""Evaluation results for the API: run summaries and a detailed breakdown of one run."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from rag.errors import NotFound
from rag.index.chunk_store import ChunkStore

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "evals" / "results"
QUESTION_FILES = ROOT / "evals"
MAX_RUNS = 30

if str(ROOT) not in sys.path:  # the evals package sits next to src/, not inside it
    sys.path.insert(0, str(ROOT))

from evals.analysis import analyse  # noqa: E402
from evals.metrics import Question, load_questions  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        k: run.get(k)
        for k in ("run", "config", "split", "retrieval_only", "questions", "config_hash", "metrics")
    }


def list_runs() -> list[dict[str, Any]]:
    if not RESULTS_DIR.exists():
        return []
    files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)[:MAX_RUNS]
    return [_summary(_load(f)) for f in files]


def _questions_for(run: dict[str, Any]) -> tuple[str | None, dict[str, Question]]:
    """The run's own question file, or (older runs) the file that matches most of its ids."""
    ids = {r["id"] for r in run["results"]}
    named = run.get("questions")
    candidates = (
        [ROOT / named]
        if named and (ROOT / named).exists()
        else [p for p in QUESTION_FILES.glob("*.jsonl") if "example" not in p.name]
    )
    best: dict[str, Question] = {}
    best_path: str | None = None
    for path in candidates:
        try:
            qs = {q.id: q for q in load_questions(path)}
        except (OSError, ValueError):
            continue
        if len(ids & qs.keys()) > len(ids & best.keys()):
            best, best_path = qs, path.relative_to(ROOT).as_posix()
    return best_path, best


def _previous(run: dict[str, Any]) -> dict[str, Any] | None:
    """The latest earlier run over the same questions and split."""
    ids = {r["id"] for r in run["results"]}
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        other = _load(path)
        if other["run"] >= run["run"] or other.get("split") != run.get("split"):
            continue
        if ids == {r["id"] for r in other["results"]}:
            return other
    return None


def _title(chunks: dict[str, Any], cid: str | None) -> str | None:
    c = chunks.get(cid) if cid else None
    return f"{c.title} > {c.section_path}" if c else None


def run_detail(run_id: str, store: ChunkStore) -> dict[str, Any]:
    path = RESULTS_DIR / f"{Path(run_id).name}.json"
    if not path.exists():
        raise NotFound(f"evaluation run not found: {run_id}")
    run = _load(path)
    questions_file, questions = _questions_for(run)
    detail = analyse(run, questions, _previous(run))
    detail["questions_file"] = run.get("questions") or questions_file
    ids = {m["first"] for m in detail["misses"]} | {g for m in detail["misses"] for g in m["gold"]}
    chunks = store.get_many(i for i in ids if i)
    for m in detail["misses"]:
        m["first_title"] = _title(chunks, m["first"])
        m["gold_title"] = _title(chunks, m["gold"][0] if m["gold"] else None)
    return detail
