"""Content-hash ledger: decides which files to add, update or delete."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from rag.index.chunk_store import ChunkStore
from rag.ingest.loaders import SUPPORTED_SUFFIXES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class IngestPlan:
    added: list[Path] = field(default_factory=list)
    changed: list[Path] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)  # doc_ids
    unchanged: int = 0
    skipped: list[str] = field(default_factory=list)


def scan(corpus_dir: Path, max_file_bytes: int, under: Path | None = None) -> list[Path]:
    root = under or corpus_dir
    if not root.exists():
        return []
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    return [
        p
        for p in files
        if p.suffix.lower() in SUPPORTED_SUFFIXES and p.stat().st_size <= max_file_bytes
    ]


def plan(
    store: ChunkStore, corpus_dir: Path, max_file_bytes: int, under: Path | None = None
) -> IngestPlan:
    files = scan(corpus_dir, max_file_bytes, under)
    result = IngestPlan()
    seen: set[str] = set()
    for path in files:
        rel = path.relative_to(corpus_dir).as_posix()
        seen.add(rel)
        record = store.document_by_path(rel)
        if record is None:
            result.added.append(path)
        elif record.sha256 != sha256_file(path):
            result.changed.append(path)
        else:
            result.unchanged += 1
    scope = (under or corpus_dir).resolve()
    for record in store.documents():
        in_scope = (corpus_dir / record.path).resolve().is_relative_to(scope)
        if in_scope and record.path not in seen:
            result.deleted.append(record.doc_id)
    return result
