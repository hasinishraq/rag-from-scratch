"""Text -> unit-length vectors. The local backend runs sentence-transformers on CPU; the optional
Ollama backend (FR-20) sends text through the gateway, like every other model call."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]


def normalise(vectors: npt.ArrayLike) -> Vectors:
    arr = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed_passages(self, texts: Sequence[str]) -> Vectors: ...

    def embed_query(self, text: str) -> npt.NDArray[np.float32]: ...


class LocalEmbedder:
    def __init__(
        self, model: str, query_prefix: str = "", passage_prefix: str = "", batch_size: int = 32
    ) -> None:
        self._model_name = model
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._batch_size = batch_size
        self._model: Any = None

    @property
    def model_id(self) -> str:
        return f"local:{self._model_name}"

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    def _encode(self, texts: list[str]) -> Vectors:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        raw = self._load().encode(
            texts, batch_size=self._batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return normalise(raw)

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        return self._encode([self._passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        return self._encode([self._query_prefix + text])[0]


class OllamaEmbedder:
    """`embed_fn` is Gateway.embed bound to the Ollama provider."""

    def __init__(
        self, model: str, embed_fn: Callable[[Sequence[str]], list[list[float]]], batch: int = 32
    ) -> None:
        self._model_name = model
        self._embed_fn = embed_fn
        self._batch = batch

    @property
    def model_id(self) -> str:
        return f"ollama:{self._model_name}"

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            out.extend(self._embed_fn(texts[start : start + self._batch]))
        return normalise(out) if out else np.zeros((0, 0), dtype=np.float32)

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        return normalise(self._embed_fn([text]))[0]
