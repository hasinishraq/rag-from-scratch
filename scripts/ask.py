"""Ask a question from the command line.

Usage: python scripts/ask.py "How long do refunds take?" [--retrieve-only] [--mode auto]
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from rag.config import load_config
from rag.errors import RagError
from rag.pipeline import Pipeline
from rag.services import build_services
from rag.types import Mode

_PREVIEW_CHARS = 240


def main() -> int:
    # Windows consoles default to cp1252; model output often contains other characters.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--retrieve-only", action="store_true", help="skip generation (no quota)")
    ap.add_argument("--mode", default=None, choices=[m.value for m in Mode])
    ap.add_argument("--top-k", type=int, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        pipeline = Pipeline(build_services(load_config(a.config)))
        ans = pipeline.answer(
            a.question, a.top_k, None, Mode(a.mode) if a.mode else None, a.retrieve_only
        )
    except RagError as exc:
        print(f"error ({exc.code}): {exc.message}")
        return 1
    if not a.retrieve_only:
        print(textwrap.fill(ans.answer, 100))
        via = f"{ans.provider} / {ans.model}" if ans.provider else "no model call"
        print(
            f"\n[{via}, {ans.latency_ms} ms{', cached' if ans.cached else ''}, "
            f"trace {ans.trace_id}]"
        )
        for w in ans.warnings:
            print(f"warning: {w}")
    print(f"\nSources ({len(ans.sources)} of {ans.n_candidates} candidates):")
    for s in ans.sources:
        mark = "*" if s.cited else " "
        where = f"{s.path}" + (f" p.{s.page}" if s.page else "")
        score = f"{s.score:.3f}" if s.score is not None else "-"
        print(f"{mark}[{s.n}] {s.title} > {s.section_path}  ({where}, score {score}, {s.chunk_id})")
        print(textwrap.indent(textwrap.shorten(s.text, _PREVIEW_CHARS), "      "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
