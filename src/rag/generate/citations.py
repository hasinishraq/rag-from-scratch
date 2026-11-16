"""Parses and validates [n] citation markers against the sources that were supplied."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from rag.generate.prompts import NOT_IN_SOURCES

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Some models (gpt-oss) write citations with CJK or fullwidth brackets: 【2】 ［2］.
_ALT_MARKER = re.compile(r"[【［〔](\d+(?:\s*[,，]\s*\d+)*)[】］〕]")


def normalise_markers(text: str) -> str:
    return _ALT_MARKER.sub(lambda m: "[" + m.group(1).replace("，", ",") + "]", text)


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_MIN_SENTENCE_CHARS = 20


@dataclass(frozen=True)
class CitationCheck:
    cited: tuple[int, ...]
    unknown: tuple[int, ...]
    uncited_sentences: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.unknown


def markers(text: str) -> list[int]:
    out: list[int] = []
    for match in _MARKER.finditer(normalise_markers(text)):
        out.extend(int(n) for n in re.split(r"\s*,\s*", match.group(1)))
    return out


def is_refusal(text: str) -> bool:
    cleaned = text.strip().strip(".").strip("`*\"' ")
    return cleaned == NOT_IN_SOURCES or (
        cleaned.startswith(NOT_IN_SOURCES) and not markers(cleaned)
    )


class RefusalGate:
    """Holds back streamed text while it could still turn out to be the refusal sentinel, so a
    reader never sees NOT_IN_SOURCES flash by before the refusal text replaces it."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._held = ""
        self._open = False

    def feed(self, delta: str) -> None:
        if self._open:
            self._emit(delta)
            return
        self._held += delta
        probe = self._held.strip().strip("`*\"' ")
        if NOT_IN_SOURCES.startswith(probe) or probe.startswith(NOT_IN_SOURCES):
            return
        self._open = True
        self._emit(self._held)


def check(text: str, n_sources: int) -> CitationCheck:
    text = normalise_markers(text)
    found = markers(text)
    cited = tuple(sorted({n for n in found if 1 <= n <= n_sources}))
    unknown = tuple(sorted({n for n in found if not 1 <= n <= n_sources}))
    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    uncited = tuple(s for s in sentences if len(s) >= _MIN_SENTENCE_CHARS and not _MARKER.search(s))
    return CitationCheck(cited=cited, unknown=unknown, uncited_sentences=uncited)
