"""Turns RepliQA's questions into an evaluation set for this app (evals/repliqa_questions.jsonl).

Run after `scripts/ingest.py --config configs/repliqa.yaml`, and again whenever the chunking
settings change (gold chunk ids depend on them). For each answerable question, the gold chunks
are the chunks of its own document that contain RepliQA's supporting paragraph (`long_answer`),
matched by 5-word phrases so PDF line breaks and hyphenation do not matter. Unanswerable
questions get no gold chunks and `answerable: false`.

Usage: python scripts/build_repliqa_eval.py [--config configs/repliqa.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from rag.config import load_config
from rag.db import Database
from rag.index.chunk_store import ChunkStore

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "repliqa" / "questions_raw.jsonl"
OUT = ROOT / "evals" / "repliqa_questions.jsonl"
SHINGLE = 5
GOLD_MIN = 0.3  # share of the supporting paragraph's phrases a chunk must contain
FALLBACK_MIN = 0.15  # accept the single best chunk above this when none reaches GOLD_MIN
TEST_SHARE = 4  # every 4th question (by stable hash) goes to the held-out test split


def words(text: str) -> list[str]:
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text.lower())  # undo hyphenation at line breaks
    return re.findall(r"[a-z0-9]+", text)


def shingles(text: str) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i : i + SHINGLE]) for i in range(max(1, len(w) - SHINGLE + 1))}


def is_unanswerable(q: dict) -> bool:
    """The card marks these with answer "UNANSWERABLE", but newer splits write a sentence
    instead; `long_answer` is "NA" exactly when there is no answer, in every split."""
    return q.get("long_answer", "").strip() == "NA" or q.get("answer") == "UNANSWERABLE"


def split_for(question_id: str) -> str:
    h = int(hashlib.sha1(question_id.encode()).hexdigest(), 16)
    return "test" if h % TEST_SHARE == 0 else "dev"


def gold_chunks(long_answer: str, chunks: dict[str, str]) -> list[str]:
    target = shingles(long_answer)
    if not target:
        return []
    scored = sorted(
        ((len(target & shingles(t)) / len(target), cid) for cid, t in chunks.items()), reverse=True
    )
    gold = [cid for score, cid in scored if score >= GOLD_MIN]
    if not gold and scored and scored[0][0] >= FALLBACK_MIN:
        gold = [scored[0][1]]
    return gold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/repliqa.yaml")
    a = ap.parse_args()
    if not RAW.exists():
        print(f"{RAW} not found: run scripts/fetch_repliqa.py first.")
        return 2
    cfg = load_config(a.config)
    store = ChunkStore(Database(cfg.store_dir / "rag.db"))
    raw = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line]
    out: list[dict] = []
    missing_doc = unmatched = 0
    for q in raw:
        answerable = not is_unanswerable(q)
        gold: list[str] = []
        if answerable:
            record = store.document_by_path(q["path"])
            if record is None:
                missing_doc += 1
                continue
            ids = store.chunk_ids_for_doc(record.doc_id)
            chunks = {cid: c.text for cid, c in store.get_many(ids).items()}
            gold = gold_chunks(q["long_answer"], chunks)
            if not gold:
                unmatched += 1
                continue
        out.append(
            {
                "id": q["question_id"],
                "question": q["question"],
                "gold_chunk_ids": gold,
                "reference_answer": "" if not answerable else q["answer"],
                "answerable": answerable,
                "type": "factual" if answerable else "unanswerable",
                "split": split_for(q["question_id"]),
                "topic": q["topic"],
                "source": "RepliQA (ServiceNow, CC BY 4.0)",
            }
        )
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    n_un = sum(not r["answerable"] for r in out)
    n_test = sum(r["split"] == "test" for r in out)
    print(
        f"wrote {len(out)} questions to {OUT} ({n_un} unanswerable; dev {len(out) - n_test}, "
        f"test {n_test}). Skipped: {missing_doc} with no indexed document, {unmatched} whose "
        "supporting paragraph matched no chunk."
    )
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
