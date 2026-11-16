from rag.config import RoutingConfig
from rag.llm.routing import Router
from rag.types import Provider, ProviderKind
from tests.fixtures.fakes import FakeClock


def _p(name: str) -> Provider:
    return Provider(
        name=name,
        kind=ProviderKind.OPENAI_COMPATIBLE,
        base_url="https://x.test",
        model="m",
        free=True,
    )


CHAIN = [_p("a"), _p("b"), _p("c")]


def _names(chain: list[Provider]) -> list[str]:
    return [p.name for p in chain]


def test_circuit_opens_after_threshold_and_recovers_after_pause() -> None:
    clock = FakeClock()
    router = Router(RoutingConfig(failure_threshold=3, pause_s=60), clock)
    for _ in range(2):
        router.failure("a", "timeout")
    assert _names(router.order(CHAIN)) == ["a", "b", "c"]  # below the threshold
    router.failure("a", "timeout")
    assert _names(router.order(CHAIN)) == ["b", "c", "a"]  # paused: tried last, not dropped
    assert router.snapshot()["a"]["paused"] is True
    clock.sleep(61)
    assert _names(router.order(CHAIN)) == ["a", "b", "c"]  # half-open: back in its place
    router.success("a", 500)
    assert router.snapshot()["a"]["consecutive_failures"] == 0


def test_success_resets_failure_streak() -> None:
    router = Router(RoutingConfig(failure_threshold=2), FakeClock())
    router.failure("a", "x")
    router.success("a", 100)
    router.failure("a", "x")
    assert not router.paused("a")


def test_rate_limit_cooldown_uses_the_providers_wait() -> None:
    clock = FakeClock()
    router = Router(RoutingConfig(pause_s=60), clock)
    router.cooldown("b", 15, "rate-limited")
    assert _names(router.order(CHAIN)) == ["a", "c", "b"]
    clock.sleep(16)
    assert not router.paused("b")


def test_fastest_strategy_orders_by_moving_average() -> None:
    router = Router(RoutingConfig(ewma_alpha=0.5), FakeClock())
    router.success("a", 3000)
    router.success("b", 400)
    # c is not measured yet, so it goes first to learn its speed.
    assert _names(router.order(CHAIN, "fastest")) == ["c", "b", "a"]
    assert _names(router.order(CHAIN, "ordered")) == ["a", "b", "c"]
    router.success("b", 1000)  # EWMA: 0.5 * 1000 + 0.5 * 400
    assert router.snapshot()["b"]["avg_latency_ms"] == 700
