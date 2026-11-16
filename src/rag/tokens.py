"""Token counting. tiktoken is an estimate for non-OpenAI models; the simple tokenizer is the
offline fallback (tiktoken downloads its encoding file on first use)."""

from __future__ import annotations

import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)

_PIECE = re.compile(r"\s*\S+|\s+")


class Tokenizer(Protocol):
    name: str

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def count(self, text: str) -> int: ...


class SimpleTokenizer:
    """Whitespace-attached words as tokens; decode(encode(x)) == x."""

    name = "simple"

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._pieces: list[str] = []

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _PIECE.findall(text):
            idx = self._vocab.get(piece)
            if idx is None:
                idx = len(self._pieces)
                self._vocab[piece] = idx
                self._pieces.append(piece)
            ids.append(idx)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self._pieces[i] for i in ids)

    def count(self, text: str) -> int:
        return len(_PIECE.findall(text))


class TiktokenTokenizer:
    def __init__(self, encoding: str) -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, disallowed_special=())

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def count(self, text: str) -> int:
        return len(self.encode(text))


def get_tokenizer(name: str) -> Tokenizer:
    if name == "simple":
        return SimpleTokenizer()
    try:
        return TiktokenTokenizer(name)
    except Exception as exc:  # offline first run, unknown encoding
        logger.warning("tiktoken encoding %r unavailable (%s); using simple tokenizer", name, exc)
        return SimpleTokenizer()
