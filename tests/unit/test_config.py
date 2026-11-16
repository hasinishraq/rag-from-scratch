from pathlib import Path

import pytest

from rag.config import load_config
from rag.errors import ConfigError

ROOT = Path(__file__).parents[2]


def test_default_config_loads() -> None:
    cfg = load_config(ROOT / "configs/default.yaml", root=ROOT)
    assert cfg.chunking.chunk_tokens == 350
    assert cfg.chunking.overlap_tokens == 50
    assert cfg.retrieval.rrf_k == 60
    assert [p.name for p in cfg.llm.providers][:2] == ["groq", "gemini"]
    assert all(p.free for p in cfg.llm.providers)


def test_extends_merges_sections() -> None:
    cfg = load_config(ROOT / "configs/hybrid.yaml", root=ROOT)
    assert cfg.rerank.enabled is False
    assert cfg.rerank.k_context == 6  # inherited
    assert cfg.retrieval.use_bm25 is True


def test_invalid_config_fails_with_clear_message(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        f"extends: {(ROOT / 'configs/default.yaml').as_posix()}\n"
        "chunking: {chunk_tokens: 50, overlap_tokens: 80}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="overlap_tokens must be smaller"):
        load_config(bad)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        f"extends: {(ROOT / 'configs/default.yaml').as_posix()}\nretrieval: {{k_retreive: 5}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"retrieval.k_retreive"):
        load_config(bad)


def test_missing_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("nope.yaml")
