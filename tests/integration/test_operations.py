"""Operations: health probes, metrics, request ids, body limit, busy limit, trace search,
retention, feedback and removing a document."""

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from rag.api import create_app
from rag.db import Database
from rag.observability import JsonFormatter, Metrics, request_id_var
from rag.types import Trace

Q = "Within how many days of approval are approved refunds issued?"
OFF_TOPIC = "Which telescope first photographed the rings of Neptune and its moons?"


def _client(env: Any) -> TestClient:
    return TestClient(create_app(env.services))


def _trace(trace_id: str, created_at: str, question: str = "q") -> Trace:
    return Trace(
        trace_id=trace_id,
        question=question,
        candidates=(),
        prompt="",
        answer="",
        refused=False,
        provider=None,
        model=None,
        tokens_in=0,
        tokens_out=0,
        latency_ms=1,
        cached=False,
        config_hash="h",
        mode="auto",
        created_at=created_at,
    )


def test_liveness_readiness_and_health(make_env: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("RAG_PRELOAD", "0")
    c = _client(make_env())
    assert c.get("/health/live").json() == {"status": "ok"}
    ready = c.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"
    assert ready.json()["checks"]["database"] is True
    health = c.get("/api/health").json()
    assert health["version"] and health["uptime_s"] >= 0 and health["ready"] is True


def test_not_ready_until_the_local_models_are_loaded(make_env: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("RAG_PRELOAD", "1")
    # Without entering the lifespan the warm-up never runs, which is the "still starting" state.
    c = _client(make_env())
    ready = c.get("/health/ready")
    assert ready.status_code == 503 and ready.json()["status"] == "starting"
    assert c.get("/health/live").status_code == 200


def test_request_id_is_returned_and_quoted_in_errors(make_env: Any) -> None:
    c = _client(make_env())
    r = c.get("/sources/nope", headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123" and r.json()["request_id"] == "abc-123"
    assert "app;dur=" in r.headers["Server-Timing"]
    generated = c.get("/health/live", headers={"X-Request-ID": "bad id with spaces"})
    assert generated.headers["X-Request-ID"] != "bad id with spaces"
    assert len(generated.headers["X-Request-ID"]) == 16


def test_metrics_endpoint_counts_by_route_template(make_env: Any) -> None:
    c = _client(make_env())
    c.get("/traces/one")
    c.get("/traces/two")
    text = c.get("/metrics").text
    assert 'rag_http_requests_total{method="GET",route="/traces/{trace_id}",status="404"} 2' in text
    assert (
        'rag_http_request_duration_seconds_bucket{route="/traces/{trace_id}",le="+Inf"} 2' in text
    )
    assert "rag_passages_indexed 0" in text and "/traces/one" not in text


def test_metrics_histogram_is_cumulative() -> None:
    m = Metrics()
    m.observe("GET", "/x", 200, 0.07)
    m.observe("GET", "/x", 200, 3.0)
    text = m.render({})
    assert '_bucket{route="/x",le="0.05"} 0' in text
    assert '_bucket{route="/x",le="0.1"} 1' in text
    assert '_bucket{route="/x",le="5.0"} 2' in text
    assert '_count{route="/x"} 2' in text


def test_json_log_lines_carry_the_request_id() -> None:
    token = request_id_var.set("req-9")
    try:
        record = logging.LogRecord("rag.x", logging.INFO, __file__, 1, "indexed %d", (3,), None)
        record.request_id = request_id_var.get()
        entry = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert entry["message"] == "indexed 3" and entry["request_id"] == "req-9"
    assert entry["level"] == "INFO" and entry["logger"] == "rag.x"


def test_oversized_body_is_refused_before_it_is_read(make_env: Any) -> None:
    c = _client(make_env(limits={"max_file_mb": 0.001}))
    big = "x" * (2 * 1024 * 1024)
    r = c.post("/documents/upload", json={"filename": "a.md", "content_base64": big})
    assert r.status_code == 413 and r.json()["code"] == "payload_too_large"


def test_busy_when_every_query_slot_is_taken(make_env: Any) -> None:
    env = make_env(limits={"max_concurrent_queries": 1})
    env.mark_tested("groq")
    c = _client(env)
    c.post("/ingest", json={})
    entered, release = threading.Event(), threading.Event()
    real = env.services.reranker.score

    def slow(question: str, texts: Any) -> list[float]:
        entered.set()
        release.wait(timeout=10)
        return real(question, texts)

    env.services.reranker.score = slow
    first: dict[str, Any] = {}
    t = threading.Thread(target=lambda: first.update(r=c.post("/query", json={"question": Q})))
    t.start()
    assert entered.wait(timeout=10)
    second = c.post("/query", json={"question": Q})
    release.set()
    t.join(timeout=10)
    assert second.status_code == 429 and second.json()["code"] == "busy"
    assert first["r"].status_code == 200
    assert c.post("/query", json={"question": Q}).status_code == 200  # the slot was given back


def test_trace_search_filter_and_paging(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq")
    c = _client(env)
    c.post("/ingest", json={})
    c.post("/query", json={"question": Q})
    c.post("/query", json={"question": OFF_TOPIC})
    c.post("/query", json={"question": Q, "retrieve_only": True})
    everything = c.get("/traces")
    assert everything.headers["X-Total-Count"] == "3" and len(everything.json()) == 3
    assert [t["question"] for t in c.get("/traces?status=refused").json()] == [OFF_TOPIC]
    assert len(c.get("/traces?status=answered").json()) == 1
    assert len(c.get("/traces?status=lookup").json()) == 1
    found = c.get("/traces", params={"q": "telescope"})
    assert found.headers["X-Total-Count"] == "1" and found.json()[0]["question"] == OFF_TOPIC
    assert c.get("/traces", params={"q": "100%_literal"}).json() == []
    page = c.get("/traces?limit=1&offset=2")
    assert len(page.json()) == 1 and page.headers["X-Total-Count"] == "3"
    assert c.get("/traces?status=nonsense").status_code == 400
    light = c.get("/traces?summary=true&status=answered").json()[0]
    assert "candidates" not in light and "prompt" not in light and light["question"] == Q


def test_feedback_is_stored_changed_and_cleared(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq")
    c = _client(env)
    c.post("/ingest", json={})
    trace_id = c.post("/query", json={"question": Q}).json()["trace_id"]
    r = c.put(f"/traces/{trace_id}/feedback", json={"rating": "down", "note": " wrong page "})
    assert r.status_code == 200
    trace = c.get(f"/traces/{trace_id}").json()
    assert trace["feedback"] == "down" and trace["feedback_note"] == "wrong page"
    assert c.get("/stats").json()["today"]["unhelpful"] == 1
    assert [t["trace_id"] for t in c.get("/traces?status=rated").json()] == [trace_id]
    c.put(f"/traces/{trace_id}/feedback", json={"rating": None})
    cleared = c.get(f"/traces/{trace_id}").json()
    assert cleared["feedback"] is None and cleared["feedback_note"] == ""
    assert c.put("/traces/missing/feedback", json={"rating": "up"}).status_code == 404
    assert c.put(f"/traces/{trace_id}/feedback", json={"rating": "meh"}).status_code == 422


def test_trace_retention_drops_old_and_surplus_rows(make_env: Any) -> None:
    store = make_env().services.chunks
    now = datetime.now()
    old = (now - timedelta(days=120)).isoformat(timespec="seconds")
    store.write_trace(_trace("old", old))
    for i in range(5):
        stamp = (now - timedelta(minutes=5 - i)).isoformat(timespec="seconds")
        store.write_trace(_trace(f"new{i}", stamp))
    cutoff = (now - timedelta(days=90)).isoformat(timespec="seconds")
    assert store.prune_traces(cutoff, max_rows=3) == 3
    assert [t.trace_id for t in store.list_traces()] == ["new4", "new3", "new2"]


def test_database_uses_wal_and_migrates_feedback_columns(tmp_path: Path) -> None:
    db = Database(tmp_path / "rag.db")
    row = db.one("PRAGMA journal_mode")
    assert row is not None and row[0] == "wal"
    cols = {r[1] for r in db.query("PRAGMA table_info(traces)")}
    assert {"feedback", "feedback_note"} <= cols
    db.close()


def test_removing_a_document_deletes_file_and_passages(make_env: Any) -> None:
    env = make_env()
    c = _client(env)
    c.post("/ingest", json={})
    docs = c.get("/documents").json()
    target = next(d for d in docs["documents"] if d["path"].endswith("refund_policy.md"))
    before = docs["chunks"]
    r = c.delete(f"/documents/{target['doc_id']}")
    assert r.status_code == 200 and r.json()["deleted"] == target["path"]
    assert not (env.cfg.corpus_dir / target["path"]).exists()
    after = c.get("/documents").json()
    assert after["chunks"] == before - target["n_chunks"]
    assert all(d["doc_id"] != target["doc_id"] for d in after["documents"])
    hits = c.post("/query", json={"question": Q, "retrieve_only": True}).json()["sources"]
    assert all(s["path"] != target["path"] for s in hits)
    # Nothing comes back on the next indexing run, and a second delete is a clean 404.
    assert c.post("/ingest", json={}).json() == {
        "added": 0,
        "updated": 0,
        "deleted": 0,
        "chunks": 0,
        "errors": [],
    }
    assert c.delete(f"/documents/{target['doc_id']}").status_code == 404
