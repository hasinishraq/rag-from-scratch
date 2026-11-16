"""Settings store: providers (order, limits, last test result), model mode and Ollama options.
Seeded from the config file on first run; edited through the API afterwards. Never holds a key."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag.config import ProviderSeed
from rag.db import Database
from rag.errors import Conflict, InvalidRequest, NotFound
from rag.llm.providers import is_local_url
from rag.llm.secrets import SecretStore
from rag.types import Limits, Mode, Provider, ProviderKind, TestResult, TestStatus

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
OLLAMA_NAME = "ollama"
_SEEDED = "seeded"
_MODE = "mode"
EMBEDDING_BACKEND = "embedding_backend"
_CONNECTION_FIELDS = frozenset(
    {"base_url", "model", "api_key", "api_key_env", "free", "allow_remote"}
)


class ProviderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: ProviderKind = ProviderKind.OPENAI_COMPATIBLE
    base_url: str
    model: str = ""
    api_key: str | None = Field(default=None, repr=False)
    api_key_env: str = ""
    limits: Limits = Limits()
    free: bool = False
    enabled: bool = True
    allow_remote: bool = False


class ProviderUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    model: str | None = None
    api_key: str | None = Field(default=None, repr=False)
    api_key_env: str | None = None
    limits: Limits | None = None
    free: bool | None = None
    enabled: bool | None = None
    position: int | None = None
    allow_remote: bool | None = None


def _row_to_provider(row: Any) -> Provider:
    return Provider(
        name=row["name"],
        kind=ProviderKind(row["kind"]),
        base_url=row["base_url"],
        model=row["model"],
        api_key_env=row["key_ref"],
        limits=Limits(**json.loads(row["limits_json"])),
        free=bool(row["free"]),
        enabled=bool(row["enabled"]),
        position=row["position"],
        allow_remote=bool(row["allow_remote"]),
        test_status=TestStatus(row["test_status"]),
        test_stage=row["test_stage"],
        test_detail=row["test_detail"],
        test_latency_ms=row["test_latency_ms"],
        tested_at=row["tested_at"],
    )


def _validate(p: Provider) -> None:
    if not _NAME.match(p.name):
        raise InvalidRequest("name: letters, digits, '-' and '_' only, at most 40 characters")
    if not p.base_url.startswith(("http://", "https://")):
        raise InvalidRequest("base_url must start with http:// or https://")
    if p.kind is ProviderKind.OLLAMA:
        if p.name != OLLAMA_NAME:
            raise InvalidRequest(f"the Ollama provider is always named '{OLLAMA_NAME}'")
        if not is_local_url(p.base_url) and not p.allow_remote:
            raise InvalidRequest(
                "this Ollama address is not on this computer; confirm 'allow remote' to send "
                "questions and passages to it"
            )
    elif not p.base_url.startswith("https://") and not is_local_url(p.base_url):
        raise InvalidRequest("hosted providers must use https://")


class SettingsStore:
    def __init__(self, db: Database, secrets: SecretStore) -> None:
        self.db = db
        self.secrets = secrets

    # ---- seeding and simple settings ----
    def seed(self, seeds: Sequence[ProviderSeed], mode: Mode) -> None:
        if self.get_setting(_SEEDED):
            return
        for i, seed in enumerate(seeds):
            p = Provider(
                name=seed.name,
                kind=seed.kind,
                base_url=seed.base_url,
                model=seed.model,
                api_key_env=seed.api_key_env,
                limits=seed.limits,
                free=seed.free,
                enabled=seed.enabled,
                position=i,
            )
            _validate(p)
            self._write(p)
        self.set_setting(_MODE, mode.value)
        self.set_setting(_SEEDED, "1")

    def get_setting(self, key: str, default: str = "") -> str:
        row = self.db.one("SELECT value FROM settings WHERE key = ?", (key,))
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, value))

    def mode(self) -> Mode:
        return Mode(self.get_setting(_MODE, Mode.AUTO.value))

    def set_mode(self, mode: Mode) -> Mode:
        self.set_setting(_MODE, mode.value)
        return mode

    # ---- providers ----
    def list(self) -> list[Provider]:
        return [
            _row_to_provider(r)
            for r in self.db.query("SELECT * FROM providers ORDER BY position, name")
        ]

    def get(self, name: str) -> Provider:
        row = self.db.one("SELECT * FROM providers WHERE name = ?", (name,))
        if row is None:
            raise NotFound(f"provider not found: {name}")
        return _row_to_provider(row)

    def key_set(self, p: Provider) -> bool:
        return self.secrets.has(p.name, p.api_key_env)

    def _write(self, p: Provider) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO providers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                p.name,
                p.kind.value,
                p.base_url,
                p.model,
                p.api_key_env,
                p.limits.model_dump_json(exclude_none=True),
                int(p.free),
                int(p.enabled),
                p.position,
                int(p.allow_remote),
                p.test_status.value,
                p.test_stage,
                p.test_detail,
                p.test_latency_ms,
                p.tested_at,
            ),
        )

    def create(self, data: ProviderInput) -> Provider:
        if self.db.one("SELECT 1 FROM providers WHERE name = ?", (data.name,)):
            raise Conflict(f"a provider named {data.name} already exists")
        row = self.db.one("SELECT COALESCE(MAX(position), -1) + 1 AS n FROM providers")
        fields = data.model_dump(exclude={"api_key"})
        p = Provider(**fields, position=int(row["n"]) if row else 0)
        _validate(p)
        self._write(p)
        if data.api_key:
            self.secrets.set(p.name, data.api_key)
        return p

    def update(self, name: str, data: ProviderUpdate) -> Provider:
        current = self.get(name)
        changes = {  # attributes, not model_dump: nested models must stay models
            k: getattr(data, k)
            for k in data.model_fields_set
            if k not in ("api_key", "position") and getattr(data, k) is not None
        }
        # A change to how the provider is reached invalidates the last test result. Enabling,
        # reordering and editing limits do not: they do not change whether the provider works.
        reset = {
            "test_status": TestStatus.UNTESTED,
            "test_stage": None,
            "test_detail": "Settings changed, not tested yet.",
            "test_latency_ms": None,
            "tested_at": None,
        }
        touched = data.model_fields_set & _CONNECTION_FIELDS
        p = current.model_copy(update={**changes, **(reset if touched else {})})
        _validate(p)
        self._write(p)
        if data.api_key:
            self.secrets.set(name, data.api_key)
        if data.position is not None:
            self.move(name, data.position)
        return self.get(name)

    def move(self, name: str, position: int) -> list[Provider]:
        names = [p.name for p in self.list()]
        if name not in names:
            raise NotFound(f"provider not found: {name}")
        names.remove(name)
        names.insert(max(0, min(position, len(names))), name)
        with self.db.tx() as conn:
            conn.executemany(
                "UPDATE providers SET position = ? WHERE name = ?",
                [(i, n) for i, n in enumerate(names)],
            )
        return self.list()

    def delete(self, name: str) -> None:
        self.get(name)
        self.db.execute("DELETE FROM providers WHERE name = ?", (name,))
        self.secrets.delete(name)

    def record_test(self, result: TestResult) -> Provider:
        last = result.stages[-1].n if result.stages else None
        self.db.execute(
            "UPDATE providers SET test_status = ?, test_stage = ?, test_detail = ?, "
            "test_latency_ms = ?, tested_at = ? WHERE name = ?",
            (
                result.status.value,
                last,
                result.detail,
                result.latency_ms,
                result.tested_at,
                result.provider,
            ),
        )
        return self.get(result.provider)
