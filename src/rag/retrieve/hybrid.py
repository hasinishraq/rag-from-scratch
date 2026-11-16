"""Dense top-k and BM25 top-k, fused with RRF, with optional path filters."""

from __future__ import annotations

from rag.index.indexes import Indexes
from rag.retrieve.fusion import rrf
from rag.types import Candidate, Filters


class HybridRetriever:
    def __init__(self, indexes: Indexes, k_retrieve: int, rrf_k: int, use_bm25: bool) -> None:
        self.indexes = indexes
        self.k_retrieve = k_retrieve
        self.rrf_k = rrf_k
        self.use_bm25 = use_bm25

    def retrieve(self, question: str, filters: Filters | None = None) -> list[Candidate]:
        allowed = None
        if filters and filters.path_prefixes:
            allowed = self.indexes.chunks.ids_with_path_prefix(filters.path_prefixes)
            if not allowed:
                return []
        query_vec = self.indexes.embedder.embed_query(question)
        dense = [cid for cid, _ in self.indexes.vectors.search(query_vec, self.k_retrieve, allowed)]
        lexical: list[str] = []
        if self.use_bm25:
            lexical = [
                cid for cid, _ in self.indexes.bm25.search(question, self.k_retrieve, allowed)
            ]
        fused = rrf([dense, lexical] if self.use_bm25 else [dense], k=self.rrf_k)
        dense_rank = {cid: i for i, cid in enumerate(dense, start=1)}
        bm25_rank = {cid: i for i, cid in enumerate(lexical, start=1)}
        stored = self.indexes.chunks.get_many(cid for cid, _ in fused)
        out: list[Candidate] = []
        for cid, score in fused:
            chunk = stored.get(cid)
            if chunk is None:
                continue
            out.append(
                Candidate(
                    chunk_id=cid,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    title=chunk.title,
                    section_path=chunk.section_path,
                    page=chunk.page,
                    path=chunk.path,
                    dense_rank=dense_rank.get(cid),
                    bm25_rank=bm25_rank.get(cid),
                    rrf_score=score,
                )
            )
        return out
