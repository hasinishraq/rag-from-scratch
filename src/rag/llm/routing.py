"""Smart routing for the gateway: per-provider health, a circuit breaker and latency ordering.

- Circuit breaker: after `failure_threshold` consecutive failures a provider is paused for
  `pause_s`, so a broken provider no longer costs a timeout on every question. After the pause
  it is tried once more (half-open); one success closes the circuit.
- Rate-limit cooldown: a 429 whose wait is too long to sit out pauses the provider until the
  time it gave, instead of asking it again on the next question.
- Strategy "fastest": healthy providers ordered by an exponentially weighted moving average of
  their latency; a provider not measured yet goes first so its speed gets learned.
Paused providers are never dropped: they go to the end of the chain as a last resort.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from rag.config import RoutingConfig
from rag.llm.ratelimit import Clock
from rag.types import Provider


@dataclass
class Health:
    ewma_ms: float | None = None
    consecutive_failures: int = 0
    paused_until: float = 0.0
    pause_reason: str = ""
    successes: int = 0
    failures: int = 0


class Router:
    def __init__(self, cfg: RoutingConfig, clock: Clock) -> None:
        self.cfg = cfg
        self.clock = clock
        self._health: dict[str, Health] = {}
        self._lock = threading.Lock()

    def _h(self, name: str) -> Health:
        return self._health.setdefault(name, Health())

    def paused(self, name: str) -> bool:
        with self._lock:
            return self._h(name).paused_until > self.clock.time()

    def order(self, chain: list[Provider], strategy: str | None = None) -> list[Provider]:
        strategy = strategy or self.cfg.strategy
        now = self.clock.time()
        with self._lock:
            ready = [p for p in chain if self._h(p.name).paused_until <= now]
            paused = sorted(
                (p for p in chain if self._h(p.name).paused_until > now),
                key=lambda p: self._h(p.name).paused_until,
            )
            if strategy == "fastest":
                # sorted() is stable, so equal latencies keep the user's order.
                ready = sorted(ready, key=lambda p: self._h(p.name).ewma_ms or 0.0)
        return ready + paused

    def success(self, name: str, latency_ms: int) -> None:
        a = self.cfg.ewma_alpha
        with self._lock:
            h = self._h(name)
            h.ewma_ms = latency_ms if h.ewma_ms is None else a * latency_ms + (1 - a) * h.ewma_ms
            h.consecutive_failures = 0
            h.paused_until = 0.0
            h.pause_reason = ""
            h.successes += 1

    def failure(self, name: str, reason: str) -> None:
        with self._lock:
            h = self._h(name)
            h.consecutive_failures += 1
            h.failures += 1
            if h.consecutive_failures >= self.cfg.failure_threshold:
                h.paused_until = self.clock.time() + self.cfg.pause_s
                h.pause_reason = f"{h.consecutive_failures} failures in a row: {reason}"

    def cooldown(self, name: str, seconds: float | None, reason: str) -> None:
        with self._lock:
            h = self._h(name)
            h.paused_until = max(h.paused_until, self.clock.time() + (seconds or self.cfg.pause_s))
            h.pause_reason = reason

    def snapshot(self) -> dict[str, dict[str, Any]]:
        now = self.clock.time()
        with self._lock:
            return {
                name: {
                    "avg_latency_ms": None if h.ewma_ms is None else round(h.ewma_ms),
                    "paused": h.paused_until > now,
                    "paused_for_s": max(0, round(h.paused_until - now)),
                    "pause_reason": h.pause_reason if h.paused_until > now else "",
                    "consecutive_failures": h.consecutive_failures,
                    "successes": h.successes,
                    "failures": h.failures,
                }
                for name, h in self._health.items()
            }
