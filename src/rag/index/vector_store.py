"""Exact vector search over a unit-normalised float32 matrix (one .npy file plus an id list)."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

Matrix = npt.NDArray[np.float32]


class VectorStore:
    def __init__(self, dim: int | None = None) -> None:
        self.ids: list[str] = []
        self.matrix: Matrix = np.zeros((0, dim or 0), dtype=np.float32)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: Sequence[str], vectors: Matrix) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors differ in length")
        if len(ids) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(self.ids) == 0:
            self.matrix = np.zeros((0, vectors.shape[1]), dtype=np.float32)
        elif vectors.shape[1] != self.dim:
            raise ValueError(f"vector dimension {vectors.shape[1]} != index dimension {self.dim}")
        self.delete(ids)
        self.matrix = np.vstack([self.matrix, vectors])
        self.ids.extend(ids)

    def delete(self, ids: Collection[str]) -> None:
        drop = set(ids)
        if not drop or not self.ids:
            return
        keep = [i for i, cid in enumerate(self.ids) if cid not in drop]
        self.matrix = self.matrix[keep]
        self.ids = [self.ids[i] for i in keep]

    def search(
        self, query: npt.NDArray[np.float32], k: int, allowed: Collection[str] | None = None
    ) -> list[tuple[str, float]]:
        if not self.ids or k <= 0:
            return []
        scores = self.matrix @ np.asarray(query, dtype=np.float32)
        if allowed is not None:
            mask = np.fromiter(
                (cid in allowed for cid in self.ids), dtype=bool, count=len(self.ids)
            )
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(self.ids))
        top = np.argpartition(-scores, k - 1)[:k]
        # Stable tie-break on row index so the order equals a naive sorted loop.
        order = sorted(top.tolist(), key=lambda i: (-float(scores[i]), i))
        return [(self.ids[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def row_map(self) -> dict[str, int]:
        return {cid: i for i, cid in enumerate(self.ids)}

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.matrix)
        (directory / "vector_ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> VectorStore:
        store = cls()
        mat_path, ids_path = directory / "vectors.npy", directory / "vector_ids.json"
        if mat_path.exists() and ids_path.exists():
            store.matrix = np.load(mat_path).astype(np.float32)
            store.ids = json.loads(ids_path.read_text(encoding="utf-8"))
        return store
