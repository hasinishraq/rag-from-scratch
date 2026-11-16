"""Okapi BM25, written out: IDF(q) * f(q,D)(k1+1) / (f(q,D) + k1(1 - b + b|D|/avgdl))."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Collection, Iterable
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: dict[str, Counter[str]] = {}
        self.lengths: dict[str, int] = {}
        self.df: Counter[str] = Counter()
        self._total_len = 0

    def __len__(self) -> int:
        return len(self.docs)

    def add(self, items: Iterable[tuple[str, str]]) -> None:
        for doc_id, text in items:
            self.remove([doc_id])
            terms = Counter(tokenize(text))
            self.docs[doc_id] = terms
            self.lengths[doc_id] = sum(terms.values())
            self._total_len += self.lengths[doc_id]
            self.df.update(terms.keys())

    def remove(self, ids: Iterable[str]) -> None:
        for doc_id in ids:
            terms = self.docs.pop(doc_id, None)
            if terms is None:
                continue
            self._total_len -= self.lengths.pop(doc_id)
            self.df.subtract(terms.keys())
            for term in terms:
                if self.df[term] <= 0:
                    del self.df[term]

    def idf(self, term: str) -> float:
        n, df = len(self.docs), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str, allowed: Collection[str] | None = None) -> dict[str, float]:
        if not self.docs:
            return {}
        avgdl = self._total_len / len(self.docs) or 1.0
        q_terms = [t for t in set(tokenize(query)) if t in self.df]
        out: dict[str, float] = {}
        for doc_id, terms in self.docs.items():
            if allowed is not None and doc_id not in allowed:
                continue
            norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / avgdl)
            score = 0.0
            for term in q_terms:
                f = terms.get(term, 0)
                if f:
                    score += self.idf(term) * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                out[doc_id] = score
        return out

    def search(
        self, query: str, k: int, allowed: Collection[str] | None = None
    ) -> list[tuple[str, float]]:
        ranked = sorted(self.scores(query, allowed).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"k1": self.k1, "b": self.b, "docs": {k: dict(v) for k, v in self.docs.items()}}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, k1: float, b: float) -> BM25Index:
        index = cls(k1=k1, b=b)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for doc_id, terms in payload["docs"].items():
                counter = Counter({t: int(n) for t, n in terms.items()})
                index.docs[doc_id] = counter
                index.lengths[doc_id] = sum(counter.values())
                index._total_len += index.lengths[doc_id]
                index.df.update(counter.keys())
        return index
