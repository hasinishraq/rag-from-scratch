"""Indexing as a background job, so the UI can show progress instead of waiting on one request.

One job at a time: a second start while one runs is refused. The state is kept in memory; the
index itself is saved by the ingestion code as before.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from rag.errors import Conflict, RagError
from rag.ingest.runner import Progress
from rag.types import IngestReport

logger = logging.getLogger(__name__)
# Below this share done, a remaining-time estimate is noise (model loading dominates).
_ETA_MIN_SHARE = 0.05

RunIngest = Callable[[str | None, bool, Progress], IngestReport]


@dataclass
class IngestJob:
    id: str
    status: str = "running"  # running | done | failed
    phase: str = "starting"  # starting | scanning | reading | embedding | saving | done
    done: int = 0
    total: int = 0
    current: str = ""
    fraction: float = 0.0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    report: dict[str, Any] | None = None
    error: str | None = None

    def view(self, now: float) -> dict[str, Any]:
        share = (
            1.0
            if self.status == "done"
            else (min(1.0, (self.done + self.fraction) / self.total) if self.total else 0.0)
        )
        elapsed = (self.finished_at or now) - self.started_at
        eta = (
            elapsed * (1 - share) / share
            if self.status == "running" and share >= _ETA_MIN_SHARE
            else None
        )
        return {
            **asdict(self),
            "percent": round(share * 100, 1),
            "elapsed_s": round(elapsed, 1),
            "eta_s": None if eta is None else round(eta, 1),
        }


class IngestJobs:
    def __init__(self, run: RunIngest, clock: Callable[[], float] = time.time) -> None:
        self._run = run
        self._clock = clock
        self._lock = threading.Lock()
        self._job: IngestJob | None = None

    def start(self, path: str | None = None, rebuild: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._job is not None and self._job.status == "running":
                raise Conflict("indexing is already running")
            job = IngestJob(id=uuid.uuid4().hex[:12], started_at=self._clock())
            self._job = job
        threading.Thread(
            target=self._work, args=(job, path, rebuild), name="ingest", daemon=True
        ).start()
        return job.view(self._clock())

    def status(self) -> dict[str, Any] | None:
        with self._lock:
            return self._job.view(self._clock()) if self._job else None

    def _work(self, job: IngestJob, path: str | None, rebuild: bool) -> None:
        def progress(phase: str, done: int, total: int, current: str, fraction: float) -> None:
            with self._lock:
                job.phase, job.done, job.total = phase, done, total
                job.current, job.fraction = current, fraction

        try:
            report = self._run(path, rebuild, progress)
            with self._lock:
                job.report, job.status, job.phase = report.model_dump(mode="json"), "done", "done"
        except RagError as exc:
            with self._lock:
                job.status, job.error = "failed", exc.message
        except Exception as exc:  # the job thread must always end in a final state
            logger.exception("indexing failed")
            with self._lock:
                job.status, job.error = "failed", f"indexing failed: {type(exc).__name__}"
        finally:
            with self._lock:
                job.finished_at = self._clock()
