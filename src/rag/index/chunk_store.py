"""SQLite tables for documents, chunks and traces."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from rag.db import Database
from rag.types import Candidate, Chunk, Document, DocumentRecord, StoredChunk, Trace

# Trace list filters -> SQL conditions. "failed" matches the Today figures on the Traces page.
_TRACE_STATUS = {
    "answered": "refused = 0 AND provider IS NOT NULL",
    "refused": "refused = 1",
    "lookup": "provider IS NULL AND refused = 0 AND errors_json = '[]'",
    "failed": "errors_json != '[]'",
    "rated": "feedback IS NOT NULL",
}
TRACE_STATUSES = frozenset(_TRACE_STATUS)


def _trace_filter(q: str, status: str) -> tuple[str, list[Any]]:
    where, params = ["1 = 1"], []
    if q.strip():
        like = "%" + q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        where.append("(question LIKE ? ESCAPE '\\' OR answer LIKE ? ESCAPE '\\')")
        params += [like, like]
    if status in _TRACE_STATUS:
        where.append(f"({_TRACE_STATUS[status]})")
    return " AND ".join(where), params


_CHUNK_COLS = (
    "c.chunk_id, c.doc_id, c.ord, c.text, c.section_path, c.page, c.n_tokens, d.title, d.path"
)


def _stored(row: sqlite3.Row) -> StoredChunk:
    data: dict[str, Any] = dict(row)
    return StoredChunk(**data)


class ChunkStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ---- documents ----
    def documents(self) -> list[DocumentRecord]:
        rows = self.db.query("SELECT * FROM documents ORDER BY path")
        return [DocumentRecord(**dict(r)) for r in rows]

    def document_by_path(self, path: str) -> DocumentRecord | None:
        row = self.db.one("SELECT * FROM documents WHERE path = ?", (path,))
        return DocumentRecord(**dict(row)) if row else None

    def upsert_document(self, doc: Document, chunks: Sequence[Chunk]) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self.db.tx() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc.doc_id,))
            conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (doc.doc_id, doc.path, doc.sha256, doc.title, len(chunks), now),
            )
            conn.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                [
                    (c.chunk_id, c.doc_id, c.ord, c.text, c.section_path, c.page, c.n_tokens)
                    for c in chunks
                ],
            )

    def delete_document(self, doc_id: str) -> list[str]:
        ids = [
            r["chunk_id"]
            for r in self.db.query("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))
        ]
        with self.db.tx() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        return ids

    def chunk_ids_for_doc(self, doc_id: str) -> list[str]:
        rows = self.db.query("SELECT chunk_id FROM chunks WHERE doc_id = ? ORDER BY ord", (doc_id,))
        return [r["chunk_id"] for r in rows]

    # ---- chunks ----
    def get(self, chunk_id: str) -> StoredChunk | None:
        row = self.db.one(
            f"SELECT {_CHUNK_COLS} FROM chunks c JOIN documents d USING (doc_id) "
            "WHERE c.chunk_id = ?",
            (chunk_id,),
        )
        return _stored(row) if row else None

    def get_many(self, chunk_ids: Iterable[str]) -> dict[str, StoredChunk]:
        ids = list(chunk_ids)
        out: dict[str, StoredChunk] = {}
        for start in range(0, len(ids), 500):  # SQLite parameter limit
            part = ids[start : start + 500]
            marks = ",".join("?" * len(part))
            rows = self.db.query(
                f"SELECT {_CHUNK_COLS} FROM chunks c JOIN documents d USING (doc_id) "
                f"WHERE c.chunk_id IN ({marks})",
                part,
            )
            out.update({r["chunk_id"]: _stored(r) for r in rows})
        return out

    def ids_with_path_prefix(self, prefixes: Sequence[str]) -> set[str]:
        out: set[str] = set()
        for prefix in prefixes:
            rows = self.db.query(
                "SELECT c.chunk_id FROM chunks c JOIN documents d USING (doc_id) "
                "WHERE d.path LIKE ? ESCAPE '\\'",
                (prefix.replace("%", "\\%").replace("_", "\\_") + "%",),
            )
            out.update(r["chunk_id"] for r in rows)
        return out

    def count(self) -> int:
        row = self.db.one("SELECT COUNT(*) AS n FROM chunks")
        return int(row["n"]) if row else 0

    def set_vec_rows(self, rows: dict[str, int]) -> None:
        with self.db.tx() as conn:
            conn.execute("UPDATE chunks SET vec_row = NULL")
            conn.executemany(
                "UPDATE chunks SET vec_row = ? WHERE chunk_id = ?",
                [(row, cid) for cid, row in rows.items()],
            )

    # ---- traces ----
    def write_trace(self, trace: Trace) -> None:
        self.db.execute(
            "INSERT INTO traces (trace_id, question, candidates_json, prompt, answer, refused, "
            "provider, model, tokens_in, tokens_out, latency_ms, cached, config_hash, mode, "
            "errors_json, created_at, timings_json, tokens_saved) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trace.trace_id,
                trace.question,
                json.dumps([c.model_dump() for c in trace.candidates]),
                trace.prompt,
                trace.answer,
                int(trace.refused),
                trace.provider,
                trace.model,
                trace.tokens_in,
                trace.tokens_out,
                trace.latency_ms,
                int(trace.cached),
                trace.config_hash,
                trace.mode,
                json.dumps(list(trace.errors)),
                trace.created_at,
                json.dumps(trace.timings),
                trace.tokens_saved,
            ),
        )

    def _trace(self, row: sqlite3.Row) -> Trace:
        data: dict[str, Any] = dict(row)
        return Trace(
            trace_id=data["trace_id"],
            question=data["question"],
            candidates=tuple(Candidate(**c) for c in json.loads(data["candidates_json"])),
            prompt=data["prompt"],
            answer=data["answer"],
            refused=bool(data["refused"]),
            provider=data["provider"],
            model=data["model"],
            tokens_in=data["tokens_in"],
            tokens_out=data["tokens_out"],
            latency_ms=data["latency_ms"],
            cached=bool(data["cached"]),
            config_hash=data["config_hash"],
            mode=data["mode"],
            errors=tuple(json.loads(data["errors_json"])),
            created_at=data["created_at"],
            timings=json.loads(data.get("timings_json") or "{}"),
            tokens_saved=data.get("tokens_saved") or 0,
            feedback=data.get("feedback"),
            feedback_note=data.get("feedback_note") or "",
        )

    def get_trace(self, trace_id: str) -> Trace | None:
        row = self.db.one("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
        return self._trace(row) if row else None

    def list_traces(
        self, limit: int = 50, offset: int = 0, q: str = "", status: str = ""
    ) -> list[Trace]:
        """Newest first. `q` searches question and answer text; `status` is a TRACE_STATUSES key."""
        where, params = _trace_filter(q, status)
        rows = self.db.query(
            f"SELECT * FROM traces WHERE {where} ORDER BY created_at DESC, rowid DESC "
            "LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        return [self._trace(r) for r in rows]

    def count_traces(self, q: str = "", status: str = "") -> int:
        where, params = _trace_filter(q, status)
        row = self.db.one(f"SELECT COUNT(*) AS n FROM traces WHERE {where}", params)
        return int(row["n"]) if row else 0

    def set_feedback(self, trace_id: str, rating: str | None, note: str = "") -> bool:
        """Stores the reader's verdict on an answer; False when the trace does not exist."""
        if self.db.one("SELECT 1 FROM traces WHERE trace_id = ?", (trace_id,)) is None:
            return False
        self.db.execute(
            "UPDATE traces SET feedback = ?, feedback_note = ? WHERE trace_id = ?",
            (rating, note if rating else "", trace_id),
        )
        return True

    def prune_traces(self, before: str, max_rows: int) -> int:
        """Retention: drops traces older than `before` and all but the newest `max_rows`."""
        n = self.count_traces()
        with self.db.tx() as conn:
            conn.execute("DELETE FROM traces WHERE created_at < ?", (before,))
            conn.execute(
                "DELETE FROM traces WHERE rowid NOT IN "
                "(SELECT rowid FROM traces ORDER BY created_at DESC, rowid DESC LIMIT ?)",
                (max_rows,),
            )
        return n - self.count_traces()

    def trace_stats(self, since: str) -> dict[str, Any]:
        """Usage since an ISO timestamp: volume, cache hits, tokens, latency percentiles."""
        row = self.db.one(
            "SELECT COUNT(*) AS queries, SUM(provider IS NOT NULL AND cached = 0) AS model_calls, "
            "SUM(cached) AS cache_hits, SUM(refused AND provider IS NULL) AS refused_free, "
            "COALESCE(SUM(CASE WHEN cached = 0 THEN tokens_in + tokens_out END), 0) AS tokens, "
            "COALESCE(SUM(CASE WHEN provider IS NOT NULL AND cached = 0 THEN tokens_saved END), 0)"
            " AS tokens_saved, "
            "SUM(provider IS NULL AND refused = 0 AND errors_json = '[]') AS lookups, "
            "SUM(errors_json != '[]') AS failed, SUM(feedback = 'up') AS helpful, "
            "SUM(feedback = 'down') AS unhelpful FROM traces WHERE created_at >= ?",
            (since,),
        )
        lat = [
            r["latency_ms"]
            for r in self.db.query(
                "SELECT latency_ms FROM traces WHERE created_at >= ? AND errors_json = '[]' "
                "ORDER BY latency_ms",
                (since,),
            )
        ]
        stats: dict[str, Any] = {
            k: (row[k] or 0) if row else 0
            for k in (
                "queries",
                "model_calls",
                "cache_hits",
                "refused_free",
                "tokens",
                "tokens_saved",
                "failed",
                "lookups",
                "helpful",
                "unhelpful",
            )
        }
        calls = stats["model_calls"]
        stats["tokens_per_call"] = round(stats["tokens"] / calls) if calls else 0
        stats["latency_p50_ms"] = lat[len(lat) // 2] if lat else None
        stats["latency_p95_ms"] = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None
        return stats
