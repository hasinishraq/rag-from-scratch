"""The three stores kept in step: vectors, BM25 statistics and the SQLite chunk table."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rag.errors import ConfigError
from rag.index.bm25 import BM25Index
from rag.index.chunk_store import ChunkStore
from rag.index.embedder import Embedder
from rag.index.vector_store import VectorStore
from rag.types import Chunk, Document

logger = logging.getLogger(__name__)
# Chunks embedded per step when progress is reported; small enough for smooth progress on CPU.
EMBED_PROGRESS_BATCH = 16


class Indexes:
    def __init__(
        self, store_dir: Path, chunks: ChunkStore, embedder: Embedder, k1: float, b: float
    ) -> None:
        self.store_dir = store_dir
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = VectorStore.load(store_dir)
        self.bm25 = BM25Index.load(store_dir / "bm25.json", k1=k1, b=b)
        self._meta_path = store_dir / "index_meta.json"
        self._check_embedder()

    def _check_embedder(self) -> None:
        if not self._meta_path.exists() or len(self.vectors) == 0:
            return
        meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
        if meta.get("embedder") != self.embedder.model_id:
            raise ConfigError(
                f"Index was built with {meta.get('embedder')} but the embedder is now "
                f"{self.embedder.model_id}. Vectors from different models are not comparable: "
                "run a full re-index (scripts/ingest.py --rebuild)."
            )

    def indexed_with(self) -> str | None:
        if not self._meta_path.exists():
            return None
        return json.loads(self._meta_path.read_text(encoding="utf-8")).get("embedder")

    def remove_document(self, doc_id: str) -> None:
        ids = self.chunks.delete_document(doc_id)
        self.vectors.delete(ids)
        self.bm25.remove(ids)

    def put_document(
        self,
        doc: Document,
        chunks: list[Chunk],
        on_batch: Callable[[int, int], None] | None = None,
        batch_size: int = EMBED_PROGRESS_BATCH,
    ) -> None:
        """Embeds and stores one document. Embedding runs in batches so `on_batch(done, total)`
        can report progress inside a long document."""
        old = self.chunks.chunk_ids_for_doc(doc.doc_id)
        self.vectors.delete(old)
        self.bm25.remove(old)
        texts = [c.embed_text for c in chunks]
        parts = []
        for start in range(0, len(texts), batch_size):
            parts.append(self.embedder.embed_passages(texts[start : start + batch_size]))
            if on_batch is not None:
                on_batch(min(start + batch_size, len(texts)), len(texts))
        vecs = np.vstack(parts) if parts else np.zeros((0, 0), dtype=np.float32)
        self.chunks.upsert_document(doc, chunks)
        ids = [c.chunk_id for c in chunks]
        self.vectors.add(ids, vecs)
        self.bm25.add((c.chunk_id, c.embed_text) for c in chunks)

    def clear(self) -> None:
        for record in self.chunks.documents():
            self.remove_document(record.doc_id)
        self.vectors = VectorStore()

    def save(self) -> None:
        self.vectors.save(self.store_dir)
        self.bm25.save(self.store_dir / "bm25.json")
        self.chunks.set_vec_rows(self.vectors.row_map())
        meta = {"embedder": self.embedder.model_id, "dim": self.vectors.dim}
        self._meta_path.write_text(json.dumps(meta), encoding="utf-8")
        logger.info("index saved: %d chunks", len(self.vectors))
