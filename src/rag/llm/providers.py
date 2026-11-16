"""Provider registry helpers: URLs, payloads, response parsing and error classification for the
OpenAI-compatible chat API (used for hosted providers and for Ollama's /v1 endpoint). No HTTP here:
the gateway sends the requests."""

from __future__ import annotations

import ipaddress
import json
import re
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

from rag.errors import (
    ModelNotFound,
    ProviderAuthError,
    ProviderError,
    RateLimited,
    TransientProviderError,
)
from rag.types import ChatRequest, Provider, ProviderKind

_DAILY = re.compile(r"per.?day|daily|\bRPD\b|\bTPD\b|PerDay", re.IGNORECASE)
_TRY_AGAIN = re.compile(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)\s*s", re.IGNORECASE)
_MODEL_MISSING = re.compile(
    r"model.*(not found|does not exist|not exist|unknown|decommission)", re.IGNORECASE
)
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})
_MAX_ERROR_CHARS = 300


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if host in _LOCAL_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def chat_base(provider: Provider) -> str:
    base = provider.base_url.rstrip("/")
    return f"{base}/v1" if provider.kind is ProviderKind.OLLAMA else base


def chat_url(provider: Provider) -> str:
    return f"{chat_base(provider)}/chat/completions"


def models_url(provider: Provider) -> str:
    return f"{chat_base(provider)}/models"


def ollama_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/api/{endpoint}"


def auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def chat_payload(model: str, request: ChatRequest, stream: bool = False) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": stream,
    }


def parse_stream_line(line: str) -> tuple[str, dict[str, Any] | None]:
    """One server-sent-events line of a streamed chat reply -> (text delta, usage if present).
    Lines that carry neither (comments, keep-alives, [DONE], reasoning-only chunks) give ("", None).
    """
    if not line.startswith("data:"):
        return "", None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return "", None
    try:
        body = json.loads(data)
    except ValueError:
        return "", None
    if not isinstance(body, dict):
        return "", None
    usage = body.get("usage") or (body.get("x_groq") or {}).get("usage")
    choices = body.get("choices") or [{}]
    delta = (choices[0].get("delta") or {}).get("content") or ""
    return str(delta), usage if isinstance(usage, dict) else None


def parse_chat(body: dict[str, Any]) -> tuple[str, int | None, int | None]:
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("provider returned an unexpected response shape") from exc
    text = message.get("content") or ""
    usage = body.get("usage") or {}
    return str(text).strip(), usage.get("prompt_tokens"), usage.get("completion_tokens")


def normalise_model_id(model_id: str) -> str:
    return model_id.removeprefix("models/")


def parse_model_list(body: dict[str, Any]) -> list[str]:
    items = body.get("data") or body.get("models") or []
    ids = [str(m.get("id") or m.get("name") or "") for m in items if isinstance(m, dict)]
    return sorted({normalise_model_id(i) for i in ids if i})


def retry_after_seconds(headers: dict[str, str], body: str) -> float | None:
    value = headers.get("retry-after")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                from datetime import UTC, datetime

                return max(0.0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError):
                pass
    match = _TRY_AGAIN.search(body)
    if match:
        return float(match.group(1) or 0) * 60 + float(match.group(2))
    return None


def short(body: str) -> str:
    return " ".join(body.split())[:_MAX_ERROR_CHARS]


def classify_error(status: int, headers: dict[str, str], body: str, name: str) -> ProviderError:
    if status == 429:
        return RateLimited(
            f"{name}: rate-limited ({short(body)})",
            retry_after_s=retry_after_seconds(headers, body),
            daily=bool(_DAILY.search(body)),
        )
    if status in (401, 403):
        return ProviderAuthError(f"{name}: the provider refused the API key (HTTP {status})")
    if status == 404 or (status == 400 and _MODEL_MISSING.search(body)):
        return ModelNotFound(f"{name}: model not found ({short(body)})")
    if status >= 500:
        return TransientProviderError(f"{name}: server error HTTP {status}")
    return ProviderError(f"{name}: HTTP {status} ({short(body)})")
