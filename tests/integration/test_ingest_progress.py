"""Indexing progress: callback order, background jobs, and the API that the UI polls."""

import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.errors import Conflict, IngestError
from rag.ingest.jobs import IngestJobs
from rag.types import IngestReport


def test_progress_reports_every_file_and_never_goes_back(make_env: Any) -> None:
    env = make_env()
    events: list[tuple[str, int, int, str, float]] = []
    report = env.pipeline.ingest(progress=lambda *e: events.append(e))
    phases = [e[0] for e in events]
    assert phases[0] == "scanning" and phases[-2:] == ["saving", "done"]
    assert {"reading", "embedding"} <= set(phases)
    files = {e[3] for e in events if e[0] == "embedding"}
    assert len(files) == report.added == 3
    shares = [(d + f) / t for _, d, t, _, f in events if t]
    assert shares == sorted(shares) and shares[-1] == 1.0 and max(shares) <= 1.0


def _wait(jobs: IngestJobs, status: str) -> dict[str, Any]:
    for _ in range(200):
        job = jobs.status()
        if job and job["status"] == status:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job never reached {status}: {jobs.status()}")


def test_job_runs_in_background_and_refuses_a_second_start() -> None:
    release = threading.Event()

    def run(path: Any, rebuild: bool, progress: Any) -> IngestReport:
        progress("embedding", 1, 4, "b.pdf", 0.5)
        release.wait(5)
        return IngestReport(added=4, updated=0, deleted=0, chunks=40)

    jobs = IngestJobs(run)
    jobs.start()
    for _ in range(200):
        if (jobs.status() or {}).get("phase") == "embedding":
            break
        time.sleep(0.01)
    running = jobs.status()
    assert running is not None and running["percent"] == 37.5 and running["current"] == "b.pdf"
    with pytest.raises(Conflict):
        jobs.start()
    release.set()
    done = _wait(jobs, "done")
    assert done["percent"] == 100.0 and done["report"]["added"] == 4
    assert done["eta_s"] is None


def test_failed_job_ends_in_a_final_state() -> None:
    def run(path: Any, rebuild: bool, progress: Any) -> IngestReport:
        raise IngestError("broken.pdf: cannot open PDF")

    jobs = IngestJobs(run)
    jobs.start()
    failed = _wait(jobs, "failed")
    assert failed["error"] == "broken.pdf: cannot open PDF" and failed["finished_at"]
    jobs.start()  # a new job may start after a failure


def test_api_job_flow(make_env: Any) -> None:
    env = make_env()
    c = TestClient(create_app(env.services))
    assert c.get("/api/ingest/jobs/current").json() == {"job": None}
    job: dict[str, Any] = {}
    started = c.post("/api/ingest/jobs", json={})
    assert started.status_code == 202 and started.json()["status"] == "running"
    for _ in range(300):
        job = c.get("/api/ingest/jobs/current").json()["job"]
        if job["status"] != "running":
            break
        time.sleep(0.02)
    assert job["status"] == "done" and job["percent"] == 100.0
    assert job["report"]["added"] == 3
    assert c.get("/api/documents").json()["chunks"] > 0
