"""Fallback, quota, free-only and caching behaviour of the gateway against the fake server."""

from typing import Any

import pytest

from rag.errors import NoProviderAvailable, ProviderNotFree, QuotaExhausted
from rag.llm.settings import ProviderInput, ProviderUpdate
from rag.types import ChatMessage, ChatRequest, Limits, Mode
from tests.conftest import GEMINI, GROQ, default_server
from tests.fixtures.fakes import HostBehaviour

Q = "Within how many days of approval are approved refunds issued?"


def _req(text: str = "Reply with the word OK.") -> ChatRequest:
    return ChatRequest(messages=(ChatMessage(role="user", content=text),), max_tokens=8)


def test_same_call_works_against_two_providers(make_env: Any) -> None:
    env = make_env()
    gw = env.services.gateway
    for name in ("groq", "gemini"):
        resp = gw.call_provider(env.services.settings.get(name), _req(), timeout=5)
        assert resp.text == "OK" and resp.provider == name


def test_429_triggers_fallback(make_env: Any) -> None:
    server = default_server()
    server.hosts[GROQ] = HostBehaviour(mode="rate_limited", models=["big-model"], retry_after="60")
    env = make_env(server=server)
    env.mark_tested("groq", "gemini")
    env.pipeline.ingest()
    ans = env.pipeline.answer(Q)
    assert ans.provider == "gemini" and "14 days" in ans.answer
    assert len(server.chat_requests(GROQ)) == 1  # Retry-After above max_wait: no retry


def test_retry_after_is_honoured(make_env: Any) -> None:
    server = default_server()
    server.hosts[GROQ] = HostBehaviour(
        mode="rate_limited", models=["big-model"], retry_after="2", fail_times=1
    )
    env = make_env(server=server)
    env.mark_tested("groq")
    resp = env.services.gateway.chat(_req(), Mode.AUTO)
    assert resp.provider == "groq"
    assert 2 in env.clock.sleeps
    assert len(server.chat_requests(GROQ)) == 2


def test_server_error_backoff_then_fallback(make_env: Any) -> None:
    server = default_server()
    server.hosts[GROQ] = HostBehaviour(mode="server_error", models=["big-model"])
    env = make_env(server=server)
    env.mark_tested("groq", "gemini")
    resp = env.services.gateway.chat(_req())
    assert resp.provider == "gemini"
    assert len(server.chat_requests(GROQ)) == env.cfg.llm.max_attempts
    assert len([s for s in env.clock.sleeps if s > 0]) == env.cfg.llm.max_attempts - 1


def test_primary_key_removed_falls_back(make_env: Any) -> None:
    env = make_env(env={"GEMINI_API_KEY": "good-key"})
    env.mark_tested("groq", "gemini")
    env.pipeline.ingest()
    ans = env.pipeline.answer(Q)
    assert ans.provider == "gemini" and not ans.refused
    assert env.server.chat_requests(GROQ) == []


def test_non_free_provider_is_refused_before_any_request(make_env: Any) -> None:
    env = make_env()
    s = env.services.settings
    p = s.update("groq", ProviderUpdate(free=False))
    with pytest.raises(ProviderNotFree):
        env.services.gateway.call_provider(p, _req(), timeout=5)
    s.db.execute("UPDATE providers SET test_status = 'working'")
    s.update("gemini", ProviderUpdate(enabled=False))
    s.update("ollama", ProviderUpdate(enabled=False))
    s.db.execute("UPDATE providers SET test_status = 'working'")
    with pytest.raises(NoProviderAvailable):
        env.services.gateway.chat(_req())
    assert env.server.requests == []


def test_user_added_provider_needs_free_confirmation(make_env: Any) -> None:
    env = make_env()
    s = env.services.settings
    p = s.create(
        ProviderInput(
            name="paid",
            base_url=f"https://{GEMINI}/v1beta/openai",
            model="lite-model",
            api_key="good-key",
            free=False,
        )
    )
    with pytest.raises(ProviderNotFree):
        env.services.gateway.call_provider(p, _req(), timeout=5)
    assert env.server.requests == []


def test_daily_limit_marks_provider_exhausted(make_env: Any) -> None:
    server = default_server()
    server.hosts[GROQ] = HostBehaviour(mode="daily_limit", models=["big-model"])
    env = make_env(server=server)
    env.mark_tested("groq", "gemini")
    assert env.services.gateway.chat(_req("a")).provider == "gemini"
    assert env.services.limiter.usage_today("groq")[2] is True
    assert env.services.gateway.chat(_req("b")).provider == "gemini"
    assert len(server.chat_requests(GROQ)) == 1  # not asked again today


def test_all_quota_exhausted_raises_quota_error(make_env: Any) -> None:
    env = make_env()
    s = env.services.settings
    s.update("groq", ProviderUpdate(limits=Limits(rpd=0)))
    s.update("gemini", ProviderUpdate(limits=Limits(rpd=0)))
    s.db.execute("UPDATE providers SET test_status = 'working' WHERE name != 'ollama'")
    with pytest.raises(QuotaExhausted):
        env.services.gateway.chat(_req(), Mode.FREE_API_ONLY)
    assert env.server.requests == []


def test_auto_skips_untested_and_free_api_only_skips_ollama(make_env: Any) -> None:
    env = make_env()
    gw = env.services.gateway
    assert gw.chain(Mode.AUTO) == []
    env.mark_tested("ollama", "gemini")
    assert [p.name for p in gw.chain(Mode.AUTO)] == ["gemini", "ollama"]
    assert [p.name for p in gw.chain(Mode.FREE_API_ONLY)] == ["groq", "gemini"]
    assert [p.name for p in gw.chain(Mode.OLLAMA_ONLY)] == ["ollama"]
