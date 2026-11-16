from pathlib import Path
from typing import Any

import pytest

from rag.db import Database
from rag.errors import Conflict, InvalidRequest
from rag.llm.secrets import SecretStore
from rag.llm.settings import ProviderInput, ProviderUpdate, SettingsStore
from rag.types import Limits, Mode, ProviderKind, TestStatus


def test_secret_lookup_order(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("FROM_FILE=file-key\n# comment\n", encoding="utf-8")
    store = SecretStore(tmp_path / "secrets.json", env={"MY_KEY": "env-key"}, dotenv=dotenv)
    assert store.resolve("p", "MY_KEY") == "env-key"
    assert store.resolve("p", "FROM_FILE") == "file-key"
    assert store.resolve("p", "") is None
    store.set("p", "saved-key")
    assert store.resolve("p", "MY_KEY") == "saved-key"  # a key saved in the app wins
    store.delete("p")
    assert store.resolve("p", "MY_KEY") == "env-key"


def _settings(tmp_path: Path) -> SettingsStore:
    return SettingsStore(
        Database(tmp_path / "s.db"), SecretStore(tmp_path / "secrets.json", env={})
    )


def _new(name: str = "custom", **kw: object) -> ProviderInput:
    data: dict[str, object] = {
        "name": name,
        "base_url": "https://api.example.test/v1",
        "model": "m1",
        "free": True,
        "limits": Limits(rpm=10),
    }
    data.update(kw)
    return ProviderInput.model_validate(data)


def test_providers_survive_restart_and_never_store_key(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    s.create(_new(api_key="sk-secret-value-1"))
    s.set_mode(Mode.FREE_API_ONLY)
    db_bytes = (tmp_path / "s.db").read_bytes()
    assert b"sk-secret-value-1" not in db_bytes
    again = _settings(tmp_path)
    p = again.get("custom")
    assert p.model == "m1" and p.limits.rpm == 10
    assert again.key_set(p)
    assert again.mode() is Mode.FREE_API_ONLY
    assert "sk-secret-value-1" not in p.model_dump_json()


def test_update_resets_status_and_reorders(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    s.create(_new("a"))
    s.create(_new("b"))
    s.db.execute("UPDATE providers SET test_status = 'working' WHERE name = 'a'")
    p = s.update("a", ProviderUpdate(model="m2"))
    assert p.test_status is TestStatus.UNTESTED
    s.move("b", 0)
    assert [p.name for p in s.list()] == ["b", "a"]


def test_only_connection_changes_reset_status(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    s.create(_new("a"))
    s.db.execute("UPDATE providers SET test_status = 'working' WHERE name = 'a'")
    p = s.update("a", ProviderUpdate(enabled=False, position=0, limits=Limits(rpm=5)))
    assert p.test_status is TestStatus.WORKING and p.limits.rpm == 5 and not p.enabled
    changes: list[dict[str, Any]] = [
        {"api_key": "new"},
        {"base_url": "https://other.test/v1"},
        {"free": False},
    ]
    for change in changes:
        s.db.execute("UPDATE providers SET test_status = 'working' WHERE name = 'a'")
        assert s.update("a", ProviderUpdate(**change)).test_status is TestStatus.UNTESTED


def test_delete_removes_key(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    s.create(_new(api_key="k-123"))
    s.delete("custom")
    assert s.secrets.all_values() == []


def test_validation(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    s.create(_new())
    with pytest.raises(Conflict):
        s.create(_new())
    with pytest.raises(InvalidRequest, match="https"):
        s.create(_new("plain", base_url="http://api.example.test/v1"))
    with pytest.raises(InvalidRequest, match="not on this computer"):
        s.create(_new("ollama", kind=ProviderKind.OLLAMA, base_url="http://10.1.2.3:11434"))
    ok = s.create(
        _new(
            "ollama", kind=ProviderKind.OLLAMA, base_url="http://10.1.2.3:11434", allow_remote=True
        )
    )
    assert ok.allow_remote
