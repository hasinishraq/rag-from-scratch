from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rag.config import AppConfig
from rag.pipeline import Pipeline
from rag.services import Services, build_services
from rag.tokens import SimpleTokenizer
from tests.fixtures.fakes import (
    FakeClock,
    FakeEmbedder,
    FakeModelServer,
    FakeReranker,
    HostBehaviour,
)

FIXTURES = Path(__file__).parent / "fixtures"
GOOD_KEY = "good-key"
GROQ = "groq.test"
GEMINI = "gemini.test"
OLLAMA = "localhost"


def make_config(tmp_path: Path, **overrides: Any) -> AppConfig:
    corpus = tmp_path / "docs"
    if not corpus.exists():
        shutil.copytree(FIXTURES / "docs", corpus)
    data: dict[str, Any] = {
        "corpus_dir": str(corpus),
        "store_dir": str(tmp_path / "store"),
        "tokenizer": "simple",
        "chunking": {"chunk_tokens": 60, "overlap_tokens": 10},
        "embedding": {"model": "fake"},
        "rerank": {"enabled": True, "model": "fake", "min_score": 0.3, "k_context": 3},
        "llm": {
            "max_wait_s": 5,
            "backoff_base_s": 0.5,
            "providers": [
                {
                    "name": "groq",
                    "kind": "openai_compatible",
                    "free": True,
                    "base_url": f"https://{GROQ}/openai/v1",
                    "api_key_env": "GROQ_API_KEY",
                    "model": "big-model",
                    "limits": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
                },
                {
                    "name": "gemini",
                    "kind": "openai_compatible",
                    "free": True,
                    "base_url": f"https://{GEMINI}/v1beta/openai",
                    "api_key_env": "GEMINI_API_KEY",
                    "model": "lite-model",
                    "limits": {"rpm": 15, "rpd": 500},
                },
                {
                    "name": "ollama",
                    "kind": "ollama",
                    "free": True,
                    "base_url": "http://localhost:11434",
                    "model": "tiny:latest",
                },
            ],
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return AppConfig.model_validate(data)


def default_server() -> FakeModelServer:
    return FakeModelServer(
        hosts={
            GROQ: HostBehaviour(models=["big-model"]),
            GEMINI: HostBehaviour(models=["lite-model"]),
            OLLAMA: HostBehaviour(models=["tiny:latest"]),
        }
    )


class Env:
    """A complete app on fakes: no network, no quota, no model downloads."""

    def __init__(
        self,
        tmp_path: Path,
        server: FakeModelServer | None = None,
        env: dict[str, str] | None = None,
        **cfg: Any,
    ) -> None:
        self.tmp_path = tmp_path
        self.server = server or default_server()
        self.clock = FakeClock()
        self.env = {"GROQ_API_KEY": GOOD_KEY, "GEMINI_API_KEY": GOOD_KEY} if env is None else env
        self.cfg = make_config(tmp_path, **cfg)
        self.services: Services = build_services(
            self.cfg,
            client=self.server.client(),
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
            clock=self.clock,
            env=self.env,
            tokenizer=SimpleTokenizer(),
            dotenv=None,
        )
        self.pipeline = Pipeline(self.services)

    def mark_tested(self, *names: str) -> None:
        for name in names:
            self.services.db.execute(
                "UPDATE providers SET test_status = 'working' WHERE name = ?", (name,)
            )


@pytest.fixture
def make_env(tmp_path: Path) -> Callable[..., Env]:
    def factory(**kwargs: Any) -> Env:
        return Env(tmp_path, **kwargs)

    return factory
