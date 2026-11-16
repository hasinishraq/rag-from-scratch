"""Typed configuration loader. Every pipeline default lives in configs/default.yaml and here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rag.errors import ConfigError
from rag.types import Limits, Mode, ProviderKind


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChunkingConfig(_Section):
    strategy: Literal["structure", "fixed"] = "structure"
    chunk_tokens: int = Field(default=350, gt=0)
    overlap_tokens: int = Field(default=50, ge=0)
    add_context_header: bool = True

    @model_validator(mode="after")
    def _overlap_below_size(self) -> ChunkingConfig:
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        return self


class EmbeddingConfig(_Section):
    backend: Literal["local", "ollama"] = "local"
    model: str
    query_prefix: str = ""
    passage_prefix: str = ""
    batch_size: int = Field(default=32, gt=0)
    ollama_model: str = ""


class RetrievalConfig(_Section):
    k_retrieve: int = Field(default=30, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    use_bm25: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


class RerankConfig(_Section):
    enabled: bool = True
    model: str = ""
    k_rerank: int = Field(default=30, gt=0)
    k_context: int = Field(default=6, gt=0)
    min_score: float = 0.2
    # Token saving: a passage scoring below this is not sent to the model (the best one always is).
    min_source_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _model_when_enabled(self) -> RerankConfig:
        if self.enabled and not self.model:
            raise ValueError("rerank.model is required when rerank.enabled is true")
        return self


class GenerationConfig(_Section):
    # Token saving: drop the overlap repeated between consecutive passages of the same file.
    trim_overlap: bool = True
    max_context_tokens: int = Field(default=2200, gt=0)
    max_output_tokens: int = Field(default=400, gt=0)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    refusal_text: str = "The documents do not contain the answer to this question."


class InputLimits(_Section):
    max_question_chars: int = Field(default=2000, gt=0)
    max_file_mb: float = Field(default=50, gt=0)
    # Questions answered at the same time; more are refused with "busy" rather than queued, so
    # the CPU reranker is never shared between more requests than it can serve promptly.
    max_concurrent_queries: int = Field(default=4, ge=1)


class TracingConfig(_Section):
    retention_days: int = Field(default=90, ge=1)  # older traces are deleted at start
    max_rows: int = Field(default=20000, ge=1)  # and only this many of the newest are kept


class HealthcheckConfig(_Section):
    hosted_timeout_s: float = Field(default=10, gt=0)
    local_timeout_s: float = Field(default=120, gt=0)
    reply_max_tokens: int = Field(default=8, gt=0)


class ProviderSeed(_Section):
    name: str = Field(min_length=1)
    kind: ProviderKind
    free: bool = False
    enabled: bool = True
    base_url: str
    api_key_env: str = ""
    model: str = ""
    limits: Limits = Limits()


class RoutingConfig(_Section):
    # ordered: the user's order. fastest: healthy providers by measured latency (EWMA).
    strategy: Literal["ordered", "fastest"] = "ordered"
    failure_threshold: int = Field(default=3, ge=1)  # consecutive failures that pause a provider
    pause_s: float = Field(default=60, gt=0)  # circuit-breaker pause before a provider is retried
    ewma_alpha: float = Field(default=0.3, gt=0, le=1)


class LLMConfig(_Section):
    cache: bool = True
    mode: Mode = Mode.AUTO
    request_timeout_s: float = Field(default=60, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    backoff_base_s: float = Field(default=1.0, ge=0)
    max_wait_s: float = Field(default=20, ge=0)
    routing: RoutingConfig = RoutingConfig()
    providers: tuple[ProviderSeed, ...] = ()

    @model_validator(mode="after")
    def _unique_names(self) -> LLMConfig:
        names = [p.name for p in self.providers]
        if len(names) != len(set(names)):
            raise ValueError("llm.providers names must be unique")
        return self


class AppConfig(_Section):
    corpus_dir: Path
    store_dir: Path
    tokenizer: str = "cl100k_base"
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    rerank: RerankConfig
    generation: GenerationConfig = GenerationConfig()
    limits: InputLimits = InputLimits()
    tracing: TracingConfig = TracingConfig()
    healthcheck: HealthcheckConfig = HealthcheckConfig()
    llm: LLMConfig = LLMConfig()

    def hash(self) -> str:
        """Hash of everything that changes an answer; recorded in each trace."""
        relevant = self.model_dump(mode="json", exclude={"llm", "limits", "healthcheck", "tracing"})
        blob = json.dumps(relevant, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    if path in seen:
        raise ConfigError(f"{path}: 'extends' forms a cycle")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    base = _read_yaml((path.parent / str(parent)).resolve(), seen | {path})
    return _deep_merge(base, raw)


def load_config(path: str | Path, root: Path | None = None) -> AppConfig:
    """Load and validate a config file. Relative directories resolve against `root` (or cwd)."""
    cfg_path = Path(path).resolve()
    data = _read_yaml(cfg_path)
    try:
        cfg = AppConfig.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"{cfg_path}: invalid configuration: {problems}") from exc
    base = root or Path.cwd()
    return cfg.model_copy(
        update={
            "corpus_dir": (base / cfg.corpus_dir).resolve(),
            "store_dir": (base / cfg.store_dir).resolve(),
        }
    )
