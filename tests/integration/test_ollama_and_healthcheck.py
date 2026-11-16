"""Ollama-only mode against the fake Ollama server, and every connection-test status."""

from typing import Any

from rag.llm.settings import ProviderUpdate
from rag.types import Mode, TestStatus
from tests.conftest import GROQ, OLLAMA, default_server
from tests.fixtures.fakes import HostBehaviour

Q = "Within how many days of approval are approved refunds issued?"


def test_ollama_only_answers_with_no_other_network(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq", "gemini", "ollama")
    env.services.settings.set_mode(Mode.OLLAMA_ONLY)
    env.pipeline.ingest()
    ans = env.pipeline.answer(Q)
    assert ans.provider == "ollama" and "[" in ans.answer and not ans.refused
    hosts = {r.url.host for r in env.server.requests}
    assert hosts == {OLLAMA}
    assert all(r.url.port == 11434 for r in env.server.requests)


def test_ollama_status_and_models(make_env: Any) -> None:
    env = make_env()
    running, models = env.services.ollama.status("http://localhost:11434", 5)
    assert running and [m.name for m in models] == ["tiny:latest"]
    env.server.hosts[OLLAMA].mode = "unreachable"
    assert env.services.ollama.status("http://localhost:11434", 5) == (False, [])


def test_ollama_embedding_through_gateway(make_env: Any) -> None:
    env = make_env()
    vecs = env.services.gateway.embed(env.services.settings.get("ollama"), "embed:latest", ["a b"])
    assert len(vecs) == 1 and len(vecs[0]) > 0


def _status(env: Any, name: str) -> TestStatus:
    result = env.services.tester.test(name)
    stored = env.services.settings.get(name)
    assert stored.test_status is result.status  # result is persisted
    assert stored.tested_at is not None
    return result.status


def test_working(make_env: Any) -> None:
    env = make_env()
    result = env.services.tester.test("groq")
    assert result.status is TestStatus.WORKING
    assert [s.outcome for s in result.stages] == ["passed"] * 5
    assert result.requests_used == 3
    assert result.latency_ms is not None
    assert len(env.server.chat_requests(GROQ)) == 3
    assert env.services.limiter.usage_today("groq")[0] == 3  # counts against quota


def test_test_bypasses_cache(make_env: Any) -> None:
    env = make_env()
    env.services.tester.test("groq")
    env.services.tester.test("groq")
    assert len(env.server.chat_requests(GROQ)) == 6


def test_working_with_warning(make_env: Any) -> None:
    server = default_server()
    server.hosts[GROQ] = HostBehaviour(models=["big-model"], refuses=False)
    env = make_env(server=server)
    assert _status(env, "groq") is TestStatus.WORKING_WITH_WARNING
    assert "refus" in env.services.settings.get("groq").test_detail


def test_key_missing(make_env: Any) -> None:
    env = make_env(env={})
    assert _status(env, "groq") is TestStatus.KEY_MISSING
    assert env.server.requests == []


def test_key_rejected(make_env: Any) -> None:
    env = make_env(env={"GROQ_API_KEY": "wrong-key"})
    assert _status(env, "groq") is TestStatus.KEY_REJECTED


def test_unreachable(make_env: Any) -> None:
    server = default_server()
    server.hosts[GROQ].mode = "unreachable"
    env = make_env(server=server)
    assert _status(env, "groq") is TestStatus.UNREACHABLE


def test_ollama_not_running(make_env: Any) -> None:
    server = default_server()
    server.hosts[OLLAMA].mode = "unreachable"
    env = make_env(server=server)
    assert _status(env, "ollama") is TestStatus.UNREACHABLE
    assert "not running" in env.services.settings.get("ollama").test_detail


def test_model_not_found(make_env: Any) -> None:
    env = make_env()
    env.services.settings.update("groq", ProviderUpdate(model="retired-model"))
    assert _status(env, "groq") is TestStatus.MODEL_NOT_FOUND


def test_ollama_model_not_pulled(make_env: Any) -> None:
    env = make_env()
    env.services.settings.update("ollama", ProviderUpdate(model="missing:7b"))
    assert _status(env, "ollama") is TestStatus.MODEL_NOT_FOUND
    assert "ollama pull missing:7b" in env.services.settings.get("ollama").test_detail


def test_rate_limited(make_env: Any) -> None:
    server = default_server()
    server.hosts[GROQ] = HostBehaviour(mode="rate_limited", models=["big-model"], retry_after="30")
    env = make_env(server=server)
    result = env.services.tester.test("groq")
    assert result.status is TestStatus.RATE_LIMITED
    assert result.retry_after_s == 30


def test_untested_after_settings_change_and_not_free(make_env: Any) -> None:
    env = make_env()
    env.services.tester.test("groq")
    p = env.services.settings.update("groq", ProviderUpdate(limits=None, model="big-model"))
    assert p.test_status is TestStatus.UNTESTED
    env.services.settings.update("groq", ProviderUpdate(free=False))
    sent = len(env.server.requests)
    assert _status(env, "groq") is TestStatus.UNTESTED
    assert len(env.server.requests) == sent  # refused before any request


def test_model_list_missing_is_not_checked(make_env: Any) -> None:
    server = default_server()
    server.hosts[GROQ] = HostBehaviour(models=["big-model"], list_endpoint=False)
    env = make_env(server=server)
    result = env.services.tester.test("groq")
    assert result.stages[2].outcome == "not_checked"
    assert result.status is TestStatus.WORKING


def test_test_all_runs_in_order(make_env: Any) -> None:
    env = make_env()
    results = env.services.tester.test_all()
    assert [r.provider for r in results] == ["groq", "gemini", "ollama"]
    assert all(r.status is TestStatus.WORKING for r in results)
