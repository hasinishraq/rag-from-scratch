"""Text cleaning: whitespace, hyphenation, page numbers, and repeated PDF headers and footers."""

from __future__ import annotations

import re
from collections import Counter

from rag.types import Block, Document

_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t ]+")
# A line that appears on at least this share of pages is a running header or footer.
_REPEAT_SHARE = 0.5
_MIN_PAGES_FOR_REPEAT = 3


def normalise_text(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [_SPACES.sub(" ", ln).strip() for ln in text.splitlines()]
    joined = " ".join(ln for ln in lines if ln)
    return joined.strip()


def _repeated_lines(blocks: tuple[Block, ...]) -> set[str]:
    pages = {b.page for b in blocks if b.page is not None}
    if len(pages) < _MIN_PAGES_FOR_REPEAT:
        return set()
    per_page: Counter[str] = Counter()
    for page in pages:
        lines = {
            _SPACES.sub(" ", ln).strip()
            for b in blocks
            if b.page == page
            for ln in b.text.splitlines()
        }
        per_page.update(ln for ln in lines if ln)
    threshold = max(2, int(len(pages) * _REPEAT_SHARE))
    return {ln for ln, n in per_page.items() if n >= threshold}


def clean_document(doc: Document) -> Document:
    repeated = _repeated_lines(doc.blocks)
    cleaned: list[Block] = []
    for block in doc.blocks:
        kept = [
            ln
            for ln in block.text.splitlines()
            if _SPACES.sub(" ", ln).strip() not in repeated and not _PAGE_NUMBER.match(ln)
        ]
        text = normalise_text("\n".join(kept))
        if text:
            cleaned.append(block.model_copy(update={"text": text}))
    return doc.model_copy(update={"blocks": tuple(cleaned)})
