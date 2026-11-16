"""Operational endpoints: liveness, readiness, summary health and metrics.

- /health/live   the process answers (no dependency is touched): restart it when this fails.
- /health/ready  it can serve questions: the database answers and the local models are loaded.
- /health        summary used by the UI: passages, mode, quota, version, uptime.
- /metrics       Prometheus text format.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from importlib import metadata
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

from rag.observability import Metrics
from rag.services import Services

PACKAGE = "rag-from-scratch"


def app_version() -> str:
    try:
        return metadata.version(PACKAGE)
    except metadata.PackageNotFoundError:  # running from a source tree that was not installed
        return "0.0.0+source"


def ops_router(
    svc: Callable[[], Services], metrics: Metrics, models_loaded: Callable[[], bool]
) -> APIRouter:
    r = APIRouter()

    def _quota(s: Services) -> dict[str, Any]:
        return {p.name: s.limiter.remaining(p.name, p.limits) for p in s.settings.list()}

    @r.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @r.get("/health/ready")
    def ready() -> JSONResponse:
        checks: dict[str, Any] = {"database": False, "models_loaded": models_loaded()}
        try:
            s = svc()
            checks["database"] = s.db.one("SELECT 1 AS ok") is not None
            checks["passages"] = s.chunks.count()
            # Informational: retrieval works without a provider, generation does not.
            checks["usable_providers"] = len(s.gateway.chain(s.settings.mode()))
        except Exception as exc:  # readiness must report, never raise
            checks["error"] = type(exc).__name__
        ok = bool(checks["database"] and checks["models_loaded"])
        body = {"status": "ready" if ok else "starting", "checks": checks}
        return JSONResponse(body, status_code=200 if ok else 503)

    @r.get("/health")
    def health() -> dict[str, Any]:
        s = svc()
        return {
            "status": "ok",
            "version": app_version(),
            "uptime_s": int(time.time() - metrics.started),
            "ready": models_loaded(),
            "chunks": s.chunks.count(),
            "mode": s.settings.mode().value,
            "quota": _quota(s),
        }

    @r.get("/metrics", response_class=PlainTextResponse)
    def metrics_text() -> str:
        s = svc()
        gauges: dict[str, float] = {
            "rag_passages_indexed": s.chunks.count(),
            "rag_documents_indexed": len(s.chunks.documents()),
            "rag_traces_stored": s.chunks.count_traces(),
            "rag_uptime_seconds": int(time.time() - metrics.started),
        }
        return metrics.render(gauges)

    return r
