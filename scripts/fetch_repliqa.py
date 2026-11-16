"""Download a slice of RepliQA (ServiceNow, CC BY 4.0) as the demo and evaluation corpus.

RepliQA's documents describe made-up people, places and organisations, so a model cannot answer
from memory: every correct answer has to come from the retrieved passages. About 20% of its
questions are unanswerable from the documents, which exercises the refusal path.

Usage: python scripts/fetch_repliqa.py [--per-topic 40] [--split repliqa_4] [--topic "..."]
Writes datasets/repliqa/pdfs/<topic>/<document_id>.pdf and datasets/repliqa/questions_raw.jsonl.
Re-running downloads only what is missing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "repliqa"
DATASET = "ServiceNow/repliqa"
ROWS_API = "https://datasets-server.huggingface.co/rows"
FILE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/"
PAGE = 100  # maximum rows per request accepted by the dataset-viewer API
PAUSE_S = 1.0  # between requests, to stay under the public API rate limit
QUESTIONS_PER_DOC = 5
DEFAULT_TOPICS = ("Company Policies", "Incident Report", "Local Technology and Innovation")
ATTRIBUTION = """# RepliQA attribution

Documents and questions in this folder come from **RepliQA** by ServiceNow,
https://huggingface.co/datasets/ServiceNow/repliqa (split `{split}`),
licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Copyright (c) ServiceNow 2023-2024. Paper: arXiv:2406.11811.

The documents are fictional. Only a subset was downloaded ({n_docs} documents, topics:
{topics}); files were renamed into one folder per topic and are otherwise unchanged.
"""


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def get_json(client: httpx.Client, url: str, params: dict[str, str | int]) -> dict:
    """GET with polite pacing: the public dataset API rate-limits bursts (HTTP 429)."""
    r: httpx.Response | None = None
    for attempt in range(8):
        r = client.get(url, params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("retry-after") or min(60, 5 * 2**attempt))
            print(f"  server asked to wait {wait:.0f} s (HTTP {r.status_code})", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        time.sleep(PAUSE_S)
        return r.json()
    if r is not None:
        r.raise_for_status()
    return {}


def collect_rows(
    client: httpx.Client, split: str, topics: list[str], n_docs: int
) -> dict[str, list[dict]]:
    """Scans the split in order until every topic has `n_docs` complete documents.
    (The viewer's filter endpoint would be shorter but is not reliably available.)"""
    wanted = set(topics)
    docs: dict[str, list[str]] = {t: [] for t in topics}
    rows: dict[str, list[dict]] = {t: [] for t in topics}
    offset, total = 0, None
    while total is None or offset < total:
        data = get_json(
            client,
            ROWS_API,
            {
                "dataset": DATASET,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": PAGE,
            },
        )
        total = data.get("num_rows_total", 0)
        batch = [item["row"] for item in data.get("rows", [])]
        if not batch:
            break
        for row in batch:
            topic = row["document_topic"]
            if topic not in wanted:
                continue
            if row["document_id"] not in docs[topic]:
                if len(docs[topic]) == n_docs:
                    continue
                docs[topic].append(row["document_id"])
            rows[topic].append(row)
        offset += len(batch)
        full = all(len(docs[t]) == n_docs for t in topics)
        if full and offset % QUESTIONS_PER_DOC == 0:
            break  # documents' rows are contiguous; stop once each topic is complete
    # Keep only documents whose questions are all present.
    for t in topics:
        counts = {d: sum(r["document_id"] == d for r in rows[t]) for d in docs[t]}
        keep = {d for d, n in counts.items() if n == QUESTIONS_PER_DOC}
        rows[t] = [r for r in rows[t] if r["document_id"] in keep]
    return rows


def download_pdf(client: httpx.Client, row: dict, target: Path) -> bool:
    if target.exists() and target.stat().st_size > 0:
        return False
    r = client.get(FILE_URL + row["document_path"], timeout=60, follow_redirects=True)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise ValueError(f"{row['document_path']} is not a PDF")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(r.content)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--split", default="repliqa_4", help="newest split: least likely seen in training"
    )
    ap.add_argument("--per-topic", type=int, default=40)
    ap.add_argument("--topic", action="append", help="repeatable; default: three topics")
    a = ap.parse_args()
    topics: list[str] = a.topic or list(DEFAULT_TOPICS)
    questions: list[dict] = []
    fetched = 0
    with httpx.Client(headers={"user-agent": "rag-from-scratch-dataset-fetch"}) as client:
        by_topic = collect_rows(client, a.split, topics, a.per_topic)
        for topic in topics:
            rows = by_topic[topic]
            docs = {r["document_id"]: r for r in rows}
            print(f"{topic}: {len(docs)} documents, {len(rows)} questions")
            for doc_id, row in docs.items():
                rel = Path("pdfs") / slug(topic) / f"{doc_id}.pdf"
                fetched += download_pdf(client, row, OUT / rel)
                for r in (x for x in rows if x["document_id"] == doc_id):
                    questions.append(
                        {
                            "question_id": r["question_id"],
                            "document_id": doc_id,
                            "topic": topic,
                            "path": rel.relative_to("pdfs").as_posix(),
                            "question": r["question"],
                            "answer": r["answer"],
                            "long_answer": r["long_answer"],
                        }
                    )
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "questions_raw.jsonl").open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    n_docs = len({q["document_id"] for q in questions})
    (OUT / "ATTRIBUTION.md").write_text(
        ATTRIBUTION.format(split=a.split, n_docs=n_docs, topics=", ".join(topics)),
        encoding="utf-8",
    )
    unanswerable = sum(q["long_answer"].strip() == "NA" for q in questions)
    print(
        f"{n_docs} documents ({fetched} downloaded now), {len(questions)} questions "
        f"({unanswerable} unanswerable) in {OUT}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
