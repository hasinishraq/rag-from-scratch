"""Ollama server status and installed-model list (GET /api/tags). Generation goes through the
gateway, which treats Ollama's OpenAI-compatible /v1 endpoint like any other provider."""

from __future__ import annotations

import httpx

from rag.errors import ProviderUnreachable
from rag.llm.providers import ollama_url
from rag.types import OllamaModel

_EMBED_FAMILIES = ("bert", "nomic-bert", "xlm-roberta")


def _kind(entry: dict[str, object]) -> str:
    details = entry.get("details") or {}
    families = []
    if isinstance(details, dict):
        families = [str(f) for f in (details.get("families") or [details.get("family") or ""])]
    name = str(entry.get("name", "")).lower()
    is_embed = "embed" in name or any(f in _EMBED_FAMILIES for f in families)
    return "embeddings" if is_embed else "chat"


def same_model(a: str, b: str) -> bool:
    """Ollama treats 'name' and 'name:latest' as the same model."""

    def norm(x: str) -> str:
        return x if ":" in x else f"{x}:latest"

    return norm(a.strip()) == norm(b.strip())


class OllamaClient:
    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def list_models(self, base_url: str, timeout: float) -> list[OllamaModel]:
        try:
            resp = self._client.get(ollama_url(base_url, "tags"), timeout=timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderUnreachable(f"Ollama is not running at {base_url}") from exc
        models = resp.json().get("models") or []
        return [
            OllamaModel(
                name=str(m.get("name") or m.get("model")),
                size_bytes=int(m.get("size") or 0),
                kind=_kind(m),  # type: ignore[arg-type]
            )
            for m in models
            if isinstance(m, dict)
        ]

    def status(self, base_url: str, timeout: float) -> tuple[bool, list[OllamaModel]]:
        try:
            return True, self.list_models(base_url, timeout)
        except ProviderUnreachable:
            return False, []
