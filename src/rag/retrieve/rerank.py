"""Cross-encoder reranking on CPU. Scores are squashed to 0..1 so min_score is model-independent."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

from rag.types import Candidate


class Reranker(Protocol):
    def score(self, question: str, texts: Sequence[str]) -> list[float]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class CrossEncoderReranker:
    def __init__(self, model: str) -> None:
        self._model_name = model
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device="cpu")
        return self._model

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        import torch

        model = self._load()
        pairs = [(question, t) for t in texts]
        try:
            raw = model.predict(pairs, activation_fn=torch.nn.Identity(), show_progress_bar=False)
        except TypeError:  # older sentence-transformers names the argument differently
            raw = model.predict(pairs, activation_fct=torch.nn.Identity(), show_progress_bar=False)
        return [_sigmoid(float(x)) for x in raw]


def rerank(
    reranker: Reranker, question: str, candidates: Sequence[Candidate], k_rerank: int, keep: int
) -> list[Candidate]:
    pool = list(candidates[:k_rerank])
    scores = reranker.score(question, [c.text for c in pool])
    scored = [c.model_copy(update={"rerank_score": s}) for c, s in zip(pool, scores, strict=True)]
    scored.sort(key=lambda c: -(c.rerank_score or 0.0))
    return scored[:keep]
