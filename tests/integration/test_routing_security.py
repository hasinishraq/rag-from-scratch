"""Smart routing through the gateway, latency/token reporting, and HTTP hardening."""

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.security import install_log_redaction, redact
from rag.types import ChatMessage, ChatRequest, Mode
from tests.conftest import GROQ, default_server
from tests.fixtures.fakes import HostBehaviour

Q = "Within how many days of approval are approved refunds issued?"


def _req(text: str) -> ChatRequest:
    return ChatRequest(messages=(ChatMessage(role="user", content=text),), max_tokens=8)


def test_broken_provider_is_paused_and_not_called_again(make_env: Any) -> None:
    server = default_server()
    server.hosts[GROQ].mode = "unreachable"
    env = make_env(server=server)
    env.mark_tested("groq", "gemini")
    threshold = env.cfg.llm.routing.failure_threshold
    for i in range(threshold):
        assert env.services.gateway.chat(_req(f"q{i}"), Mode.AUTO).provider == "gemini"
    calls = len([r for r in server.requests if r.url.host == GROQ])
    assert calls == threshold
    assert env.services.gateway.chat(_req("after"), Mode.AUTO).provider == "gemini"
    assert len([r for r in server.requests if r.url.host == GROQ]) == calls  # skipped now


def test_paused_provider_is_still_a_last_resort(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq", "gemini")
    router = env.services.gateway.router
    router.cooldown("gemini", 600, "test")
    router.cooldown("groq", 600, "test")
    assert env.services.gateway.chat(_req("x"), Mode.FREE_API_ONLY).provider == "groq"


def test_long_rate_limit_skips_provider_on_later_questions(make_env: Any) -> None:
    server = default_server()
    server.hosts[GROQ] = HostBehaviour(mode="rate_limited", models=["big-model"], retry_after="90")
    env = make_env(server=server)
    env.mark_tested("groq", "gemini")
    assert env.services.gateway.chat(_req("one"), Mode.AUTO).provider == "gemini"
    assert env.services.gateway.chat(_req("two"), Mode.AUTO).provider == "gemini"
    assert len(server.chat_requests(GROQ)) == 1


def test_empty_answer_falls_back(make_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env()
    env.mark_tested("groq", "gemini")
    gateway = env.services.gateway
    real = gateway._post_chat

    def post(provider: Any, key: Any, request: Any, timeout: float) -> Any:
        text, t_in, t_out = real(provider, key, request, timeout)
        return ("", t_in, t_out) if provider.name == "groq" else (text, t_in, t_out)

    monkeypatch.setattr(gateway, "_post_chat", post)
    resp = gateway.chat(_req("Reply with the word OK."), Mode.AUTO)
    assert resp.provider == "gemini" and resp.text == "OK"
    assert "empty answer" in resp.attempts[0]


def test_fastest_strategy_from_settings(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq", "gemini")
    c = TestClient(create_app(env.services))
    assert c.put("/api/settings/routing", json={"strategy": "fastest"}).status_code == 200
    env.services.gateway.router.success("groq", 5000)
    env.services.gateway.router.success("gemini", 300)
    assert env.services.gateway.chat(_req("fast?"), Mode.AUTO).provider == "gemini"
    assert c.get("/api/settings").json()["routing_strategy"] == "fastest"
    assert c.put("/api/settings/routing", json={"strategy": "random"}).status_code == 422


def test_answer_reports_timings_tokens_and_stats(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq")
    env.pipeline.ingest()
    ans = env.pipeline.answer(Q)
    assert {"retrieve_ms", "rerank_ms", "generate_ms", "total_ms"} <= set(ans.timings)
    assert ans.tokens_in > 0 and ans.tokens_out > 0
    trace = env.services.chunks.get_trace(ans.trace_id)
    assert trace is not None and trace.timings == ans.timings
    env.pipeline.answer(Q)  # cached
    stats = TestClient(create_app(env.services)).get("/api/stats").json()
    today = stats["today"]
    assert today["queries"] == 2 and today["model_calls"] == 1 and today["cache_hits"] == 1
    assert today["tokens"] == ans.tokens_in + ans.tokens_out
    assert today["lookups"] == 0
    assert stats["providers"]["groq"]["successes"] == 1


def test_injection_in_a_source_is_flagged_on_the_answer(make_env: Any) -> None:
    env = make_env(rerank={"min_score": 0.0})
    env.mark_tested("groq")
    env.pipeline.ingest()
    ans = env.pipeline.answer("Which carrier ships parcels and are tracking numbers emailed?")
    assert any("looks like instructions" in w for w in ans.warnings)


def test_foreign_host_and_cross_site_writes_are_refused(make_env: Any) -> None:
    env = make_env()
    c = TestClient(create_app(env.services))
    assert c.get("/api/health", headers={"host": "evil.example"}).status_code == 400
    r = c.put(
        "/api/settings/mode", json={"mode": "auto"}, headers={"origin": "https://evil.example"}
    )
    assert r.status_code == 403 and r.json()["code"] == "forbidden_origin"
    ok = c.put(
        "/api/settings/mode", json={"mode": "auto"}, headers={"origin": "http://localhost:5173"}
    )
    assert ok.status_code == 200
    assert c.get("/api/health", headers={"origin": "https://evil.example"}).status_code == 200


def test_security_headers(make_env: Any) -> None:
    env = make_env()
    r = TestClient(create_app(env.services)).get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"


def test_keys_are_redacted_from_logs_and_errors(caplog: pytest.LogCaptureFixture) -> None:
    install_log_redaction()
    key = "gsk_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"
    caplog.set_level(logging.INFO)
    logging.getLogger("rag.test").info("calling with %s and Bearer abcdefghijklmnop", key)
    assert key not in caplog.text and "abcdefghijklmnop" not in caplog.text
    assert "[redacted]" in caplog.text
    assert redact("AIza" + "x" * 35) == "[redacted]"
    assert redact("Refunds take 14 days.") == "Refunds take 14 days."


def test_redaction_leaves_ordinary_words_and_log_record_shape_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    install_log_redaction()
    font = "/assets/schibsted-grotesk-latin-wght-normal-Bb8VGrTG.woff2"
    assert redact(font) == font
    assert redact("task-runner and risk-assessment-framework-document") == (
        "task-runner and risk-assessment-framework-document"
    )
    caplog.set_level(logging.INFO)
    # uvicorn's access formatter unpacks a 5-tuple of args; redaction must keep that shape.
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "127.0.0.1:1", "GET", font, "1.1", 200
    )
    record = caplog.records[-1]
    assert isinstance(record.args, tuple) and len(record.args) == 5
    assert font in caplog.text


def test_startup_warm_up_does_not_deadlock(make_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Starts the app like the server does: services built lazily, warm-up thread first."""
    import threading

    import rag.api as api_module

    env = make_env()
    monkeypatch.setattr(api_module, "load_config", lambda _path: env.cfg)
    monkeypatch.setattr(api_module, "build_services", lambda _cfg: env.services)
    result: dict[str, int] = {}

    def run() -> None:
        with TestClient(api_module.create_app()) as c:  # runs the lifespan (warm-up)
            result["status"] = c.get("/api/health").status_code

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=20)
    assert not t.is_alive(), "API did not answer: warm-up and request deadlocked"
    assert result["status"] == 200


def test_tokens_saved_only_counts_when_a_prompt_is_sent(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq")
    env.pipeline.ingest()
    lookup = env.pipeline.answer(Q, retrieve_only=True)
    assert lookup.tokens_saved == 0
    refused = env.pipeline.answer("What is the boiling point of mercury on Venus?")
    assert refused.refused and refused.tokens_saved == 0
