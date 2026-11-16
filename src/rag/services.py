"""Wiring: builds every component once from the config. Tests inject fakes through the arguments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from rag.config import AppConfig
from rag.db import Database
from rag.errors import ConfigError
from rag.index.chunk_store import ChunkStore
from rag.index.embedder import Embedder, LocalEmbedder, OllamaEmbedder
from rag.index.indexes import Indexes
from rag.llm.cache import ResponseCache
from rag.llm.gateway import Gateway
from rag.llm.healthcheck import ConnectionTester
from rag.llm.ollama import OllamaClient
from rag.llm.ratelimit import Clock, RateLimiter, SystemClock
from rag.llm.secrets import SecretStore
from rag.llm.settings import EMBEDDING_BACKEND, OLLAMA_NAME, SettingsStore
from rag.retrieve.rerank import CrossEncoderReranker, Reranker
from rag.tokens import Tokenizer, get_tokenizer

EMBEDDING_OLLAMA_MODEL = "embedding_ollama_model"


@dataclass
class Services:
    cfg: AppConfig
    db: Database
    tok: Tokenizer
    secrets: SecretStore
    settings: SettingsStore
    limiter: RateLimiter
    cache: ResponseCache
    gateway: Gateway
    ollama: OllamaClient
    tester: ConnectionTester
    chunks: ChunkStore
    indexes: Indexes
    reranker: Reranker | None

    def close(self) -> None:
        """Releases the HTTP connections and the database file (called once at shutdown)."""
        self.gateway.client.close()
        self.db.close()


def embedding_choice(cfg: AppConfig, settings: SettingsStore) -> tuple[str, str]:
    backend = settings.get_setting(EMBEDDING_BACKEND, cfg.embedding.backend)
    model = settings.get_setting(EMBEDDING_OLLAMA_MODEL, cfg.embedding.ollama_model)
    return backend, model


def make_embedder(cfg: AppConfig, settings: SettingsStore, gateway: Gateway) -> Embedder:
    backend, ollama_model = embedding_choice(cfg, settings)
    if backend == "ollama":
        if not ollama_model:
            raise ConfigError("embedding backend is ollama but no Ollama embedding model is set")

        def embed(texts: Any) -> list[list[float]]:
            return gateway.embed(settings.get(OLLAMA_NAME), ollama_model, texts)

        return OllamaEmbedder(ollama_model, embed, cfg.embedding.batch_size)
    e = cfg.embedding
    return LocalEmbedder(e.model, e.query_prefix, e.passage_prefix, e.batch_size)


def build_services(
    cfg: AppConfig,
    *,
    client: httpx.Client | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    clock: Clock | None = None,
    env: dict[str, str] | None = None,
    tokenizer: Tokenizer | None = None,
    dotenv: Path | None = Path(".env"),
) -> Services:
    db = Database(cfg.store_dir / "rag.db")
    tok = tokenizer or get_tokenizer(cfg.tokenizer)
    secrets = SecretStore(cfg.store_dir / "secrets.json", env=env, dotenv=dotenv)
    settings = SettingsStore(db, secrets)
    settings.seed(cfg.llm.providers, cfg.llm.mode)
    clock = clock or SystemClock()
    limiter = RateLimiter(db, clock)
    cache = ResponseCache(db)
    http = client or httpx.Client()
    gateway = Gateway(settings, secrets, limiter, cache, tok, cfg.llm, http, clock)
    ollama = OllamaClient(http)
    tester = ConnectionTester(gateway, settings, ollama, cfg.healthcheck)
    chunks = ChunkStore(db)
    emb = embedder or make_embedder(cfg, settings, gateway)
    r = cfg.retrieval
    indexes = Indexes(cfg.store_dir, chunks, emb, r.bm25_k1, r.bm25_b)
    if reranker is None and cfg.rerank.enabled:
        reranker = CrossEncoderReranker(cfg.rerank.model)
    return Services(
        cfg,
        db,
        tok,
        secrets,
        settings,
        limiter,
        cache,
        gateway,
        ollama,
        tester,
        chunks,
        indexes,
        reranker if cfg.rerank.enabled else None,
    )
