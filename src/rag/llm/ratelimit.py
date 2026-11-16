"""Sliding one-minute windows for RPM and TPM, and daily RPD/TPD counters persisted in SQLite so a
restart does not reset them. Check-and-reserve is atomic so no configured limit can be exceeded."""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import UTC, datetime
from typing import Protocol

from rag.db import Database
from rag.types import Limits

WINDOW_S = 60.0
DAY_S = 86400.0


class Clock(Protocol):
    def time(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def time(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class RateLimiter:
    def __init__(self, db: Database, clock: Clock) -> None:
        self.db = db
        self.clock = clock
        self._lock = threading.Lock()
        self._requests: dict[str, deque[float]] = {}
        self._tokens: dict[str, deque[list[float]]] = {}  # [timestamp, tokens]

    def day(self) -> str:
        return datetime.fromtimestamp(self.clock.time(), UTC).date().isoformat()

    def seconds_to_reset(self) -> float:
        return DAY_S - (self.clock.time() % DAY_S)  # counters reset at UTC midnight

    def usage_today(self, provider: str) -> tuple[int, int, bool]:
        row = self.db.one(
            "SELECT requests, tokens, exhausted FROM quota WHERE provider = ? AND day = ?",
            (provider, self.day()),
        )
        return (row["requests"], row["tokens"], bool(row["exhausted"])) if row else (0, 0, False)

    def _prune(self, provider: str, now: float) -> tuple[deque[float], deque[list[float]]]:
        reqs = self._requests.setdefault(provider, deque())
        toks = self._tokens.setdefault(provider, deque())
        while reqs and reqs[0] <= now - WINDOW_S:
            reqs.popleft()
        while toks and toks[0][0] <= now - WINDOW_S:
            toks.popleft()
        return reqs, toks

    def _minute_wait(self, provider: str, limits: Limits, est: int, now: float) -> float | None:
        reqs, toks = self._prune(provider, now)
        wait = 0.0
        if limits.rpm is not None and limits.rpm <= 0:
            return None
        if limits.rpm is not None and len(reqs) >= limits.rpm:
            wait = max(wait, reqs[len(reqs) - limits.rpm] + WINDOW_S - now)
        if limits.tpm is not None:
            if est > limits.tpm:
                return None  # can never fit in one minute
            used = sum(int(t[1]) for t in toks)
            for ts, n in toks:
                if used + est <= limits.tpm:
                    break
                used -= int(n)
                wait = max(wait, ts + WINDOW_S - now)
        return wait

    def try_acquire(self, provider: str, limits: Limits, est_tokens: int) -> float | None:
        """0.0: reserved now. >0: seconds until the minute window allows it. None: not today."""
        with self._lock:
            requests, tokens, exhausted = self.usage_today(provider)
            if exhausted:
                return None
            if limits.rpd is not None and requests + 1 > limits.rpd:
                return None
            if limits.tpd is not None and tokens + est_tokens > limits.tpd:
                return None
            now = self.clock.time()
            wait = self._minute_wait(provider, limits, est_tokens, now)
            if wait is None or wait > 0:
                return wait
            self._requests[provider].append(now)
            self._tokens[provider].append([now, float(est_tokens)])
            self._add_daily(provider, 1, est_tokens)
            return 0.0

    def settle(self, provider: str, estimated: int, actual: int) -> None:
        """Correct the reservation with the token count from the response's usage field."""
        with self._lock:
            toks = self._tokens.get(provider)
            if toks:
                toks[-1][1] = float(actual)
            self._add_daily(provider, 0, actual - estimated)

    def mark_exhausted(self, provider: str) -> None:
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO quota (provider, day, exhausted) VALUES (?, ?, 1) "
                "ON CONFLICT(provider, day) DO UPDATE SET exhausted = 1",
                (provider, self.day()),
            )

    def _add_daily(self, provider: str, requests: int, tokens: int) -> None:
        with self.db.tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO quota (provider, day) VALUES (?, ?)", (provider, self.day())
            )
            conn.execute(
                "UPDATE quota SET requests = requests + ?, tokens = MAX(0, tokens + ?) "
                "WHERE provider = ? AND day = ?",
                (requests, tokens, provider, self.day()),
            )

    def remaining(self, provider: str, limits: Limits) -> dict[str, int | bool | None]:
        requests, tokens, exhausted = self.usage_today(provider)
        return {
            "requests_today": requests,
            "tokens_today": tokens,
            "requests_left": None if limits.rpd is None else max(0, limits.rpd - requests),
            "tokens_left": None if limits.tpd is None else max(0, limits.tpd - tokens),
            "exhausted": exhausted,
        }
