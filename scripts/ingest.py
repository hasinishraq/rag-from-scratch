"""Index the corpus folder. Only new or changed files are processed; deleted files are removed.

Usage: python scripts/ingest.py --config configs/default.yaml [--path sub/folder] [--rebuild]
"""

from __future__ import annotations

import argparse
import logging
import sys

from rag.config import load_config
from rag.errors import RagError
from rag.pipeline import Pipeline
from rag.services import build_services


def main() -> int:
    # Windows consoles default to cp1252; model output often contains other characters.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--path", default=None, help="only this file or folder inside the corpus")
    ap.add_argument("--rebuild", action="store_true", help="full re-index (embedder changed)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # model-download chatter
    try:
        cfg = load_config(a.config)
        cfg.corpus_dir.mkdir(parents=True, exist_ok=True)
        report = Pipeline(build_services(cfg)).ingest(a.path, a.rebuild)
    except RagError as exc:
        print(f"error: {exc.message}")
        return 1
    print(f"corpus: {cfg.corpus_dir}")
    print(
        f"added {report.added}, updated {report.updated}, deleted {report.deleted}, "
        f"chunks written {report.chunks}"
    )
    for err in report.errors:
        print(f"skipped: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
