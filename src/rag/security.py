"""HTTP and logging hardening.

- Redaction: anything that looks like an API key is replaced before it reaches a log line or an
  error message (defence in depth; keys are never logged on purpose).
- Host check: only requests addressed to this machine's names are served, which blocks DNS
  rebinding (a web page re-pointing its own domain at 127.0.0.1 to reach this API).
- Origin check: a browser request that changes something must come from this app's own origin,
  so another website open in the same browser cannot drive the API (CSRF).
- Security headers on every response; a strict Content-Security-Policy on the web UI.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

from fastapi import Request, Response
from fastapi.responses import JSONResponse

# Each pattern must start at a token boundary, so words such as "grotesk-latin" are left alone.
_KEY_PATTERNS = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(gsk_[A-Za-z0-9]{20,}"  # Groq
    r"|sk-or-v1-[A-Za-z0-9]{20,}"  # OpenRouter
    r"|sk-(?:proj-)?[A-Za-z0-9_]{20,}"  # OpenAI-style
    r"|AIza[0-9A-Za-z_-]{30,}"  # Google
    r"|AQ\.[A-Za-z0-9_-]{30,})"  # Google (newer format)
    r"|(?<=Bearer )[A-Za-z0-9._-]{12,}"
)
REDACTED = "[redacted]"
DEFAULT_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]", "testserver")
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_UI_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
    "base-uri 'self'; form-action 'self'"
)
_DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


def redact(text: str) -> str:
    return _KEY_PATTERNS.sub(REDACTED, text)


def install_log_redaction() -> None:
    """Every log record, from any library, is formatted and redacted once at creation."""
    previous = logging.getLogRecordFactory()
    if getattr(previous, "_rag_redacting", False):
        return

    def factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = previous(*args, **kwargs)  # type: ignore[arg-type]
        # Redact message and arguments separately: formatters such as uvicorn's access log
        # unpack record.args, so its shape must not change.
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {
                k: redact(v) if isinstance(v, str) else v for k, v in record.args.items()
            }
        return record

    factory._rag_redacting = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


def allowed_hosts() -> list[str]:
    extra = [h.strip() for h in os.environ.get("RAG_ALLOWED_HOSTS", "").split(",") if h.strip()]
    return [*DEFAULT_HOSTS, *extra]


def _host_of(value: str) -> str:
    parsed = urlparse(value if "://" in value else f"//{value}")
    return (parsed.hostname or "").lower()


Handler = Callable[[Request], Awaitable[Response]]


def make_guard(hosts: list[str]) -> Callable[[Request, Handler], Awaitable[Response]]:
    allowed = {h.strip("[]").lower() for h in hosts}

    async def guard(request: Request, call_next: Handler) -> Response:
        host = _host_of(request.headers.get("host", ""))
        if host not in allowed:
            return JSONResponse(
                {"code": "forbidden_host", "message": f"host {host or '?'} is not allowed"},
                status_code=400,
            )
        origin = request.headers.get("origin")
        if request.method in _UNSAFE_METHODS and origin and _host_of(origin) not in allowed:
            return JSONResponse(
                {"code": "forbidden_origin", "message": "requests from other websites are refused"},
                status_code=403,
            )
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        is_html = response.headers.get("content-type", "").startswith("text/html")
        if is_html and not request.url.path.startswith(_DOCS_PATHS):
            response.headers.setdefault("Content-Security-Policy", _UI_CSP)
        return response

    return guard
