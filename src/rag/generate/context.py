"""Orders and numbers the reranked chunks, stopping at the context token budget.

Two token savers (both reported as `tokens_saved`):
- passages scoring below `min_score` are not sent, except the best one;
- when two consecutive chunks of one file are both sent, the text the later one repeats from
  the earlier one (the chunking overlap) is cut from the prompt. Display text is unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from rag.generate.prompts import format_source
from rag.tokens import Tokenizer
from rag.types import Candidate

_ORD = re.compile(r"_(\d+)$")
_MIN_OVERLAP_CHARS = 20  # shorter shared text is coincidence, not chunk overlap


@dataclass(frozen=True)
class Context:
    sources: list[Candidate] = field(default_factory=list)
    prompt_texts: list[str] = field(default_factory=list)  # text actually sent, per source
    tokens_saved: int = 0
    dropped_low_score: int = 0


def _ord(chunk_id: str) -> int | None:
    m = _ORD.search(chunk_id)
    return int(m.group(1)) if m else None


def trim_overlap(previous: str, text: str) -> str:
    """Removes the longest prefix of `text` that is a suffix of `previous`."""
    limit = min(len(previous), len(text))
    for k in range(limit, _MIN_OVERLAP_CHARS - 1, -1):
        if previous.endswith(text[:k]):
            return text[k:].lstrip()
    return text


def _prompt_texts(chosen: list[Candidate], trim: bool) -> list[str]:
    by_pos = {(c.doc_id, _ord(c.chunk_id)): c for c in chosen}
    out: list[str] = []
    for c in chosen:
        n = _ord(c.chunk_id)
        prev = by_pos.get((c.doc_id, n - 1)) if trim and n is not None else None
        out.append(trim_overlap(prev.text, c.text) if prev is not None else c.text)
    return out


def build_context(
    candidates: Sequence[Candidate],
    tok: Tokenizer,
    max_tokens: int,
    max_sources: int,
    min_score: float = 0.0,
    trim: bool = False,
) -> Context:
    chosen: list[Candidate] = []
    used = 0
    saved = 0
    dropped = 0
    for cand in candidates:
        if len(chosen) >= max_sources:
            break
        score = cand.rerank_score
        if chosen and score is not None and score < min_score:
            saved += tok.count(cand.text)
            dropped += 1
            continue
        cost = tok.count(format_source(len(chosen) + 1, cand.title, cand.section_path, cand.text))
        if used + cost > max_tokens:
            if chosen:
                break
            continue  # a single oversized chunk is skipped rather than truncated mid-sentence
        chosen.append(cand)
        used += cost
    texts = _prompt_texts(chosen, trim)
    saved += sum(tok.count(c.text) - tok.count(t) for c, t in zip(chosen, texts, strict=True))
    return Context(
        sources=chosen, prompt_texts=texts, tokens_saved=saved, dropped_low_score=dropped
    )
