"""Incremental ingestion: ledger plan -> load -> clean -> chunk -> the three stores."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from rag.config import AppConfig
from rag.errors import IngestError, InvalidRequest, NotFound
from rag.index.indexes import Indexes
from rag.ingest import ledger
from rag.ingest.chunkers import chunk_document
from rag.ingest.clean import clean_document
from rag.ingest.loaders import SUPPORTED_SUFFIXES, load_file
from rag.tokens import Tokenizer
from rag.types import IngestReport

logger = logging.getLogger(__name__)


def resolve_under(corpus_dir: Path, path: str | None) -> Path | None:
    """A request may narrow ingestion to a sub-path, but never escape the corpus folder."""
    if not path:
        return None
    target = (corpus_dir / path).resolve()
    if not target.is_relative_to(corpus_dir.resolve()):
        raise InvalidRequest("path must be inside the corpus folder")
    if not target.exists():
        raise InvalidRequest(f"path not found in corpus: {path}")
    return target


def save_upload(corpus_dir: Path, folder: str, filename: str, data: bytes, max_bytes: int) -> str:
    """Writes an uploaded file into the corpus; returns its corpus-relative path."""
    name = Path(filename.replace("\\", "/")).name  # drop any client-side directories
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise InvalidRequest(f"{name}: only PDF, Markdown and HTML files can be indexed")
    if len(data) > max_bytes:
        raise InvalidRequest(f"{name}: larger than {max_bytes // (1024 * 1024)} MB")
    root = corpus_dir.resolve()
    target_dir = (root / folder.strip().strip("/\\")).resolve() if folder.strip() else root
    if not target_dir.is_relative_to(root):
        raise InvalidRequest("folder must be inside the corpus folder")
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / name).write_bytes(data)
    return (target_dir / name).relative_to(root).as_posix()


def remove_file(corpus_dir: Path, indexes: Indexes, doc_id: str) -> str:
    """Deletes one indexed file from the corpus folder and its passages from all three stores;
    returns its corpus-relative path. The file goes too: left in place, the next indexing run
    would add it back."""
    record = next((d for d in indexes.chunks.documents() if d.doc_id == doc_id), None)
    if record is None:
        raise NotFound(f"document not found: {doc_id}")
    root = corpus_dir.resolve()
    target = (root / record.path).resolve()
    if not target.is_relative_to(root):  # a stored path must never point outside the corpus
        raise InvalidRequest("document path is outside the corpus folder")
    target.unlink(missing_ok=True)
    indexes.remove_document(doc_id)
    indexes.save()
    logger.info("removed %s", record.path)
    return record.path


class Progress(Protocol):
    """Receives indexing progress: a phase, files done and total, the current file, and how far
    the current file's embedding has got (0..1)."""

    def __call__(
        self, phase: str, done: int, total: int, current: str, fraction: float
    ) -> None: ...


def _no_progress(phase: str, done: int, total: int, current: str, fraction: float) -> None:
    return None


def ingest(
    cfg: AppConfig,
    indexes: Indexes,
    tok: Tokenizer,
    path: str | None = None,
    rebuild: bool = False,
    progress: Progress = _no_progress,
) -> IngestReport:
    if rebuild:
        indexes.clear()
    under = resolve_under(cfg.corpus_dir, path)
    max_bytes = int(cfg.limits.max_file_mb * 1024 * 1024)
    progress("scanning", 0, 0, "", 0.0)
    todo = ledger.plan(indexes.chunks, cfg.corpus_dir, max_bytes, under)
    for doc_id in todo.deleted:
        indexes.remove_document(doc_id)
    errors: list[str] = []
    n_chunks = 0
    files = [*todo.added, *todo.changed]
    for i, file_path in enumerate(files):
        name = file_path.relative_to(cfg.corpus_dir).as_posix()
        progress("reading", i, len(files), name, 0.0)
        try:
            doc = clean_document(load_file(file_path, cfg.corpus_dir))
        except IngestError as exc:
            errors.append(exc.message)
            logger.warning("skipped %s: %s", file_path.name, exc.message)
            continue
        ch = cfg.chunking
        chunks = chunk_document(
            doc, tok, ch.chunk_tokens, ch.overlap_tokens, ch.add_context_header, ch.strategy
        )
        indexes.put_document(
            doc,
            chunks,
            on_batch=lambda d, t, i=i, name=name: progress("embedding", i, len(files), name, d / t),
        )
        n_chunks += len(chunks)
        logger.info("indexed %s: %d chunks", doc.path, len(chunks))
    if todo.added or todo.changed or todo.deleted or rebuild:
        progress("saving", len(files), len(files), "", 0.0)
        indexes.save()
    progress("done", len(files), len(files), "", 0.0)
    return IngestReport(
        added=len(todo.added),
        updated=len(todo.changed),
        deleted=len(todo.deleted),
        chunks=n_chunks,
        errors=tuple(errors),
    )
