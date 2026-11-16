"""Request ids, access logs, log format and metrics.

- Every request gets an id (the caller's X-Request-ID when it sends a sane one). It is returned
  in the response, written on every log line of that request, and quoted in error bodies, so a
  report from the UI can be matched to the server log.
- Logs go to stderr as text, or as one JSON object per line with RAG_LOG_FORMAT=json.
- GET /metrics serves counters and latency histograms in the Prometheus text format, kept in
  memory (they restart with the process; durable numbers live in the traces table).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from fastapi import Request, Response
from fastapi.responses import JSONResponse

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
access_log = logging.getLogger("rag.access")

_SANE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_BUCKETS_S = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
_QUIET_PREFIXES = ("/assets/", "/health/live")
_HANDLER_NAME = "rag-console"
_DURATION = "rag_http_request_duration_seconds"

Handler = Callable[[Request], Awaitable[Response]]


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging() -> None:
    """Gives the application's loggers a handler once. Without it, INFO lines (indexing, warm-up,
    access log) are dropped by Python's default WARNING threshold."""
    root = logging.getLogger("rag")
    if any(h.get_name() == _HANDLER_NAME for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.set_name(_HANDLER_NAME)
    handler.addFilter(_RequestIdFilter())
    if os.environ.get("RAG_LOG_FORMAT", "text").lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(request_id)s %(name)s: %(message)s")
        )
    root.addHandler(handler)
    root.setLevel(os.environ.get("RAG_LOG_LEVEL", "INFO").upper())
    root.propagate = False


class Metrics:
    """Request counts and latency histograms per route. Thread-safe; labels are route templates
    (/traces/{trace_id}), never raw paths, so the number of series stays small."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count: dict[tuple[str, str, int], int] = {}
        self._buckets: dict[str, list[int]] = {}
        self._sum: dict[str, float] = {}
        self.started = time.time()

    def observe(self, method: str, route: str, status: int, seconds: float) -> None:
        with self._lock:
            key = (method, route, status)
            self._count[key] = self._count.get(key, 0) + 1
            buckets = self._buckets.setdefault(route, [0] * (len(_BUCKETS_S) + 1))
            for i, edge in enumerate(_BUCKETS_S):
                if seconds <= edge:
                    buckets[i] += 1
            buckets[-1] += 1
            self._sum[route] = self._sum.get(route, 0.0) + seconds

    def render(self, gauges: dict[str, float]) -> str:
        """Prometheus text exposition format, version 0.0.4."""
        out = ["# TYPE rag_http_requests_total counter"]
        with self._lock:
            for (method, route, status), n in sorted(self._count.items()):
                out.append(
                    f'rag_http_requests_total{{method="{method}",route="{route}",'
                    f'status="{status}"}} {n}'
                )
            out.append(f"# TYPE {_DURATION} histogram")
            for route, buckets in sorted(self._buckets.items()):
                label = f'route="{route}"'
                edges = [*map(str, _BUCKETS_S), "+Inf"]
                for edge, n in zip(edges, buckets, strict=True):
                    out.append(f'{_DURATION}_bucket{{{label},le="{edge}"}} {n}')
                out.append(f"{_DURATION}_sum{{{label}}} {self._sum[route]:.4f}")
                out.append(f"{_DURATION}_count{{{label}}} {buckets[-1]}")
        for name, value in sorted(gauges.items()):
            out += [f"# TYPE {name} gauge", f"{name} {value}"]
        return "\n".join(out) + "\n"


def _route_of(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    # Unmatched paths (static files, 404s) share one label instead of one series per URL.
    return path if isinstance(path, str) else "static"


def make_request_context(
    metrics: Metrics, max_body_bytes: Callable[[], int]
) -> Callable[[Request, Handler], Awaitable[Response]]:
    async def request_context(request: Request, call_next: Handler) -> Response:
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _SANE_ID.match(incoming) else uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        started = time.perf_counter()
        try:
            declared = request.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > max_body_bytes():
                response: Response = JSONResponse(
                    {
                        "code": "payload_too_large",
                        "message": "the request body is larger than this server accepts",
                        "request_id": rid,
                    },
                    status_code=413,
                )
            else:
                response = await call_next(request)
            seconds = time.perf_counter() - started
            response.headers["X-Request-ID"] = rid
            response.headers["Server-Timing"] = f"app;dur={seconds * 1000:.0f}"
            route = _route_of(request)
            metrics.observe(request.method, route, response.status_code, seconds)
            if not request.url.path.startswith(_QUIET_PREFIXES):
                access_log.info(
                    "%s %s %d %.0fms",
                    request.method,
                    request.url.path,
                    response.status_code,
                    seconds * 1000,
                )
            return response
        finally:
            request_id_var.reset(token)

    return request_context
