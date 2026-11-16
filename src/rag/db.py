"""One SQLite file shared by the chunk store, cache, quota counters, settings and traces."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, sha256 TEXT NOT NULL,
    title TEXT NOT NULL, n_chunks INTEGER NOT NULL, ingested_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, ord INTEGER NOT NULL, text TEXT NOT NULL,
    section_path TEXT NOT NULL, page INTEGER, n_tokens INTEGER NOT NULL, vec_row INTEGER);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY, response_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS quota (
    provider TEXT NOT NULL, day TEXT NOT NULL, requests INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0, exhausted INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, day));
CREATE TABLE IF NOT EXISTS providers (
    name TEXT PRIMARY KEY, kind TEXT NOT NULL, base_url TEXT NOT NULL, model TEXT NOT NULL,
    key_ref TEXT NOT NULL DEFAULT '', limits_json TEXT NOT NULL DEFAULT '{}',
    free INTEGER NOT NULL DEFAULT 0, enabled INTEGER NOT NULL DEFAULT 1,
    position INTEGER NOT NULL DEFAULT 0, allow_remote INTEGER NOT NULL DEFAULT 0,
    test_status TEXT NOT NULL DEFAULT 'untested', test_stage INTEGER, test_detail TEXT NOT NULL
    DEFAULT '', test_latency_ms INTEGER, tested_at TEXT);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY, question TEXT NOT NULL, candidates_json TEXT NOT NULL,
    prompt TEXT NOT NULL, answer TEXT NOT NULL, refused INTEGER NOT NULL, provider TEXT,
    model TEXT, tokens_in INTEGER NOT NULL, tokens_out INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL, cached INTEGER NOT NULL, config_hash TEXT NOT NULL,
    mode TEXT NOT NULL, errors_json TEXT NOT NULL, created_at TEXT NOT NULL,
    timings_json TEXT NOT NULL DEFAULT '{}', tokens_saved INTEGER NOT NULL DEFAULT 0,
    feedback TEXT, feedback_note TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS traces_created ON traces(created_at);
"""

# Columns added after the first release: (table, column, definition). Applied when missing.
MIGRATIONS = (
    ("traces", "timings_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("traces", "tokens_saved", "INTEGER NOT NULL DEFAULT 0"),
    ("traces", "feedback", "TEXT"),
    ("traces", "feedback_note", "TEXT NOT NULL DEFAULT ''"),
)
# Readers do not block the writer and a crash cannot leave a half-written page (WAL); a second
# process (a CLI run beside the server) waits for the lock instead of failing at once.
PRAGMAS = ("journal_mode = WAL", "synchronous = NORMAL", "busy_timeout = 5000")


class Database:
    """Thread-safe wrapper: FastAPI runs sync handlers in a thread pool."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            for pragma in PRAGMAS:
                self._conn.execute(f"PRAGMA {pragma}")
            self._conn.executescript(SCHEMA)
            for table, column, definition in MIGRATIONS:
                cols = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
                if column not in cols:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            self._conn.commit()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self.tx() as conn:
            conn.execute(sql, params)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
