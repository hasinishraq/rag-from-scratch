from pathlib import Path

from rag.db import Database
from rag.llm.cache import ResponseCache, cache_key
from rag.llm.ratelimit import RateLimiter
from rag.types import ChatMessage, ChatRequest, ChatResponse, Limits
from tests.fixtures.fakes import FakeClock


def _drive(limiter: RateLimiter, clock: FakeClock, limits: Limits, est: int, n: int) -> list[float]:
    """Sends n requests as fast as allowed; returns the times they were admitted."""
    admitted: list[float] = []
    while len(admitted) < n:
        wait = limiter.try_acquire("p", limits, est)
        if wait is None:
            break
        if wait > 0:
            clock.sleep(wait)
            continue
        admitted.append(clock.time())
        clock.sleep(0.1)
    return admitted


def _max_in_window(times: list[float], weights: list[int] | None = None) -> int:
    weights = weights or [1] * len(times)
    best = 0
    for i, start in enumerate(times):
        window = [w for t, w in zip(times[i:], weights[i:], strict=True) if t < start + 60]
        best = max(best, sum(window))
    return best


def test_rpm_never_exceeded(tmp_path: Path) -> None:
    clock = FakeClock()
    limiter = RateLimiter(Database(tmp_path / "q.db"), clock)
    times = _drive(limiter, clock, Limits(rpm=5), est=10, n=40)
    assert len(times) == 40
    assert _max_in_window(times) <= 5


def test_tpm_never_exceeded(tmp_path: Path) -> None:
    clock = FakeClock()
    limiter = RateLimiter(Database(tmp_path / "q.db"), clock)
    times = _drive(limiter, clock, Limits(tpm=1000), est=300, n=20)
    assert len(times) == 20
    assert _max_in_window(times, [300] * len(times)) <= 1000
    assert limiter.try_acquire("p", Limits(tpm=100), 500) is None  # can never fit


def test_daily_limits_persist_across_restart(tmp_path: Path) -> None:
    clock = FakeClock()
    db_path = tmp_path / "q.db"
    limiter = RateLimiter(Database(db_path), clock)
    limits = Limits(rpd=3, tpd=10_000)
    assert len(_drive(limiter, clock, limits, est=100, n=10)) == 3
    restarted = RateLimiter(Database(db_path), clock)
    assert restarted.try_acquire("p", limits, 100) is None
    clock.sleep(86_400)  # next UTC day
    assert restarted.try_acquire("p", limits, 100) == 0.0


def test_tpd_and_settle(tmp_path: Path) -> None:
    clock = FakeClock()
    limiter = RateLimiter(Database(tmp_path / "q.db"), clock)
    limits = Limits(tpd=1000)
    assert limiter.try_acquire("p", limits, 900) == 0.0
    limiter.settle("p", 900, 200)  # the response used fewer tokens than estimated
    assert limiter.usage_today("p")[1] == 200
    assert limiter.try_acquire("p", limits, 900) is None
    assert limiter.try_acquire("p", limits, 700) == 0.0


def test_mark_exhausted(tmp_path: Path) -> None:
    clock = FakeClock()
    limiter = RateLimiter(Database(tmp_path / "q.db"), clock)
    limiter.mark_exhausted("p")
    assert limiter.try_acquire("p", Limits(), 1) is None
    assert limiter.remaining("p", Limits())["exhausted"] is True


def _req(temp: float = 0.1, text: str = "hi") -> ChatRequest:
    return ChatRequest(messages=(ChatMessage(role="user", content=text),), temperature=temp)


def test_cache_keys() -> None:
    assert cache_key(_req()) == cache_key(_req())
    assert cache_key(_req(0.2)) != cache_key(_req())
    assert cache_key(_req(text="other")) != cache_key(_req())


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = ResponseCache(Database(tmp_path / "c.db"))
    assert cache.get(_req()) is None
    resp = ChatResponse(text="x", provider="p", model="m", tokens_in=1, tokens_out=1, latency_ms=5)
    cache.put(_req(), resp)
    hit = cache.get(_req())
    assert hit is not None and hit.cached and hit.text == "x"
