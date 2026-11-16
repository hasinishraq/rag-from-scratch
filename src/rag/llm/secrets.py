"""API keys. The only writer of data/store/secrets.json; keys never leave this module except to
the gateway. Lookup order: a key saved in the app, then the named environment variable (or the
same name in a local .env file)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)


def _restrict_permissions(path: Path) -> None:
    if sys.platform != "win32":
        os.chmod(path, 0o600)
        return
    # Windows has no chmod 600: remove inherited ACLs and grant only the current user. The grant
    # uses the SID, because a bare user name can resolve to a machine account of the same name.
    system32 = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32"
    try:  # absolute paths: a Unix 'whoami' on PATH (Git Bash) takes different arguments
        out = subprocess.run(
            [str(system32 / "whoami.exe"), "/user", "/fo", "csv", "/nh"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        sid = out.strip().split(",")[-1].strip('"')
        if not sid.startswith("S-"):
            raise OSError("unexpected whoami output")
        subprocess.run(
            [str(system32 / "icacls.exe"), str(path), "/inheritance:r", "/grant:r", f"*{sid}:F"],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("could not restrict permissions on the secrets file: %s", type(exc).__name__)


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.removeprefix("export ").partition("=")
        out[name.strip()] = value.strip().strip("'\"")
    return out


class SecretStore:
    def __init__(
        self, path: Path, env: Mapping[str, str] | None = None, dotenv: Path | None = None
    ) -> None:
        self.path = path
        self._env = env if env is not None else os.environ
        self._dotenv = dotenv
        self._lock = threading.Lock()

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, self.path)
        _restrict_permissions(self.path)

    def set(self, provider: str, key: str) -> None:
        with self._lock:
            data = self._load()
            data[provider] = key.strip()
            self._write(data)

    def delete(self, provider: str) -> None:
        with self._lock:
            data = self._load()
            if data.pop(provider, None) is not None:
                self._write(data)

    def resolve(self, provider: str, env_name: str = "") -> str | None:
        with self._lock:
            saved = self._load().get(provider)
        if saved:
            return saved
        if env_name:
            value = self._env.get(env_name) or (
                _read_dotenv(self._dotenv).get(env_name) if self._dotenv else None
            )
            return value or None
        return None

    def has(self, provider: str, env_name: str = "") -> bool:
        return self.resolve(provider, env_name) is not None

    def all_values(self) -> list[str]:
        """For the secret-scan tests only."""
        with self._lock:
            return list(self._load().values())
