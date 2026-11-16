"""Streamed answers: same result as POST /query, delivered piece by piece, with the gateway's
checks, fallback and quota accounting unchanged."""

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.errors import Cancelled, StreamInterrupted
from rag.generate.citations import RefusalGate
from rag.llm.providers import parse_stream_line
from rag.types import ChatMessage, ChatRequest, Mode, StreamEvent
from tests.conftest import GEMINI, GROQ, default_server
from tests.fixtures.fakes import HostBehaviour

Q = "Within how many days of approval are approved refunds issued?"
OFF_TOPIC = "Which telescope first photographed the rings of Neptune and its moons?"


def _req(text: str = "Reply with the word OK.") -> ChatRequest:
    return ChatRequest(messages=(ChatMessage(role="user", content=text),), max_tokens=8)


def _events(response: Any) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in response.text.split("\n\n"):
        lines = [x for x in block.splitlines() if not x.startswith(":")]
        if len(lines) == 2:
            out.append((lines[0].removeprefix("event: "), json.loads(lines[1][len("data: ") :])))
    return out


def test_parse_stream_line() -> None:
    line = 'data: {"choices":[{"delta":{"content":"Hi"}}],"usage":{"prompt_tokens":3}}'
    assert parse_stream_line(line) == ("Hi", {"prompt_tokens": 3})
    assert parse_stream_line('data: {"choices":[{"delta":{"reasoning":"hm"}}]}') == ("", None)
    assert parse_stream_line('data: {"choices":[],"x_groq":{"usage":{"completion_tokens":2}}}') == (
        "",
        {"completion_tokens": 2},
    )
    for noise in ("", ": keep-alive", "data: [DONE]", "data: not json", "event: ping"):
        assert parse_stream_line(noise) == ("", None)


def test_gateway_stream_gives_the_same_reply_and_counts_usage(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq")
    pieces: list[str] = []
    plain = env.services.gateway.chat(_req(), Mode.AUTO, use_cache=False)
    before = env.services.limiter.usage_today("groq")
    streamed = env.services.gateway.chat(_req(), Mode.AUTO, use_cache=False, on_delta=pieces.append)
    assert streamed.text == plain.text == "".join(pieces).strip() == "OK"
    assert (streamed.tokens_in, streamed.tokens_out) == (plain.tokens_in, plain.tokens_out)
    after = env.services.limiter.usage_today("groq")
    assert after[0] == before[0] + 1 and after[1] == before[1] + streamed.tokens_in + 1
    sent = json.loads(env.server.chat_requests(GROQ)[-1].content)
    assert sent["stream"] is True


def test_stream_falls_back_before_the_first_piece(make_env: Any) -> None:
    server = default_server()
    server.hosts[GROQ] = HostBehaviour(mode="daily_limit", models=["big-model"])
    env = make_env(server=server)
    env.mark_tested("groq", "gemini")
    pieces: list[str] = []
    resp = env.services.gateway.chat(_req(), Mode.AUTO, on_delta=pieces.append)
    assert resp.provider == "gemini" and "".join(pieces).strip() == "OK"
    assert resp.attempts and resp.attempts[0].startswith("groq")


def test_stream_that_drops_midway_is_not_retried_elsewhere(make_env: Any) -> None:
    server = default_server()
    server.hosts[GROQ] = HostBehaviour(models=["big-model"], break_stream=True)
    env = make_env(server=server)
    env.mark_tested("groq", "gemini")
    pieces: list[str] = []
    with pytest.raises(StreamInterrupted):
        env.services.gateway.chat(_req("one two three four"), Mode.AUTO, on_delta=pieces.append)
    assert pieces  # the reader already saw text, so a second provider must not start over
    assert not server.chat_requests(GEMINI)


def test_refusal_gate_hides_the_sentinel() -> None:
    shown: list[str] = []
    gate = RefusalGate(shown.append)
    for piece in ("NOT_", "IN_", "SOURCES"):
        gate.feed(piece)
    assert shown == []
    shown.clear()
    gate = RefusalGate(shown.append)
    for piece in ("NO", "T every ", "refund [1]."):
        gate.feed(piece)
    assert "".join(shown) == "NOT every refund [1]."


def test_pipeline_reports_sources_then_text(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq")
    env.pipeline.ingest()
    events: list[StreamEvent] = []
    answer = env.pipeline.answer(Q, on_event=events.append)
    assert events[0].kind == "sources" and len(events[0].sources) == len(answer.sources)
    text = "".join(e.text for e in events if e.kind == "delta")
    assert text.strip() == answer.answer and "[1]" in text
    assert answer.provider == "groq" and answer.tokens_out > 0


def test_answer_at_the_output_limit_carries_a_warning(make_env: Any) -> None:
    env = make_env(generation={"max_output_tokens": 3})
    env.mark_tested("groq")
    env.pipeline.ingest()
    cut = env.pipeline.answer(Q)
    assert any("output limit (3 tokens)" in w for w in cut.warnings)
    roomy = make_env(generation={"max_output_tokens": 400})
    roomy.mark_tested("groq")
    assert not any("output limit" in w for w in roomy.pipeline.answer(Q).warnings)


def test_cancelling_stops_the_answer_and_is_traced(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq")
    env.pipeline.ingest()

    def stop(event: StreamEvent) -> None:
        if event.kind == "delta":
            raise Cancelled("stopped")

    with pytest.raises(Cancelled):
        env.pipeline.answer(Q, on_event=stop)
    trace = env.services.chunks.list_traces(1)[0]
    assert trace.errors and trace.errors[0].startswith("cancelled")


def test_query_stream_endpoint(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq")
    c = TestClient(create_app(env.services))
    c.post("/ingest", json={})
    r = c.post("/api/query/stream", json={"question": Q})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = _events(r)
    kinds = [k for k, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and "delta" in kinds
    done = events[-1][1]
    assert "".join(d["text"] for k, d in events if k == "delta").strip() == done["answer"]
    assert any(s["cited"] for s in done["sources"])
    assert c.get(f"/traces/{done['trace_id']}").json()["answer"] == done["answer"]


def test_query_stream_refusal_and_error_events(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq")
    c = TestClient(create_app(env.services))
    c.post("/ingest", json={})
    events = _events(c.post("/query/stream", json={"question": OFF_TOPIC}))
    assert [k for k, _ in events] == ["done"] and events[0][1]["refused"] is True
    for name in ("groq", "gemini", "ollama"):
        c.put(f"/providers/{name}", json={"enabled": False})
    events = _events(c.post("/query/stream", json={"question": Q}))
    assert events[-1][0] == "error" and events[-1][1]["code"] == "no_provider_available"
