"""API contract: endpoints, error model, and the provider add -> test -> use flow (A10)."""

from typing import Any

from fastapi.testclient import TestClient

from rag.api import create_app
from tests.conftest import GEMINI, GROQ, default_server
from tests.fixtures.fakes import HostBehaviour

Q = "Within how many days of approval are approved refunds issued?"


def _client(env: Any) -> TestClient:
    return TestClient(create_app(env.services))


def test_health_ingest_query_sources(make_env: Any) -> None:
    env = make_env()
    env.mark_tested("groq")
    c = _client(env)
    ingest = c.post("/ingest", json={}).json()
    assert set(ingest) >= {"added", "updated", "deleted", "chunks"} and ingest["added"] == 3
    health = c.get("/health").json()
    assert health["status"] == "ok" and health["chunks"] > 0 and "groq" in health["quota"]
    r = c.post("/query", json={"question": Q})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"answer", "refused", "sources", "trace_id", "provider", "cached"}
    cited = next(s for s in body["sources"] if s["cited"])
    src = c.get(f"/sources/{cited['chunk_id']}").json()
    assert set(src) >= {"text", "title", "section_path", "page", "path"}
    assert "14 days" in src["text"]
    trace = c.get(f"/traces/{body['trace_id']}").json()
    assert trace["answer"] == body["answer"] and trace["prompt"]
    assert c.get("/documents").json()["chunks"] == health["chunks"]


def test_error_model(make_env: Any) -> None:
    env = make_env()
    c = _client(env)
    r = c.get("/sources/nope")
    assert r.status_code == 404 and r.json()["code"] == "not_found"
    r = c.post("/query", json={"question": ""})
    assert r.status_code == 422 and r.json()["code"] == "invalid_request"
    r = c.post("/query", json={"question": "x" * 3000})
    assert r.status_code == 400 and r.json()["code"] == "invalid_request"


def test_quota_exhausted_is_503(make_env: Any) -> None:
    server = default_server()
    for host in (GROQ, GEMINI):
        server.hosts[host] = HostBehaviour(mode="daily_limit", models=["big-model", "lite-model"])
    env = make_env(server=server, rerank={"min_score": 0.0})
    env.mark_tested("groq", "gemini")
    c = _client(env)
    c.post("/ingest", json={})
    r = c.post("/query", json={"question": Q, "mode": "free_api_only"})
    assert r.status_code == 503
    assert r.json()["code"] == "quota_exhausted"


def test_provider_add_test_use_from_api_alone(make_env: Any) -> None:
    env = make_env()
    c = _client(env)
    c.post("/ingest", json={})
    for name in ("groq", "gemini", "ollama"):
        c.put(f"/providers/{name}", json={"enabled": False})
    body = {
        "name": "mine",
        "base_url": f"https://{GEMINI}/v1beta/openai",
        "model": "lite-model",
        "api_key": "good-key",
        "free": True,
        "limits": {"rpm": 10, "rpd": 100},
    }
    created = c.post("/providers", json=body)
    assert created.status_code == 201
    view = created.json()
    assert view["test_status"] == "untested" and view["key_set"] is True
    assert "good-key" not in created.text
    assert c.post("/query", json={"question": Q}).status_code == 503  # untested: skipped
    test = c.post("/providers/mine/test").json()
    assert test["status"] == "working" and len(test["stages"]) == 5
    ans = c.post("/query", json={"question": Q}).json()
    assert ans["provider"] == "mine" and not ans["refused"]
    listed = c.get("/providers")
    assert "good-key" not in listed.text
    edited = c.put("/providers/mine", json={"model": "lite-model", "position": 0}).json()
    assert edited["test_status"] == "untested"
    assert c.get("/providers").json()[0]["name"] == "mine"
    assert c.get("/providers/mine/models").json()["models"] == ["lite-model"]
    assert c.delete("/providers/mine").status_code == 200
    assert c.get("/providers/mine/models").status_code == 404
    assert env.services.secrets.all_values() == []


def test_provider_requires_free_and_unique_name(make_env: Any) -> None:
    env = make_env()
    c = _client(env)
    dup = c.post("/providers", json={"name": "groq", "base_url": "https://x.test/v1"})
    assert dup.status_code == 409
    c.post(
        "/providers",
        json={
            "name": "paid",
            "base_url": f"https://{GEMINI}/v1beta/openai",
            "model": "lite-model",
            "api_key": "good-key",
            "free": False,
        },
    )
    result = c.post("/providers/paid/test").json()
    assert result["status"] == "untested" and "not marked free" in result["detail"]


def test_mode_settings_and_ollama_models(make_env: Any) -> None:
    env = make_env()
    c = _client(env)
    assert c.put("/settings/mode", json={"mode": "ollama_only"}).json() == {"mode": "ollama_only"}
    assert c.get("/settings").json()["mode"] == "ollama_only"
    assert c.put("/settings/mode", json={"mode": "paid"}).status_code == 422
    models = c.get("/ollama/models").json()
    assert models["running"] and models["models"][0]["name"] == "tiny:latest"
    remote = c.get("/ollama/models", params={"base_url": "http://192.168.1.9:11434"})
    assert remote.status_code == 400
    assert c.post("/providers/test-all").status_code == 200
    assert len(c.get("/traces").json()) == 0


def test_upload_then_index_and_api_prefix(make_env: Any) -> None:
    import base64

    env = make_env()
    c = _client(env)
    content = base64.b64encode(b"# Warranty\n\nThe warranty lasts two years.\n").decode()
    r = c.post(
        "/api/documents/upload",
        json={"filename": r"C:\fake\warranty.md", "folder": "new", "content_base64": content},
    )
    assert r.status_code == 201 and r.json()["saved"] == "new/warranty.md"
    assert c.post("/api/ingest", json={}).json()["added"] == 4
    bad = c.post("/api/documents/upload", json={"filename": "x.exe", "content_base64": content})
    assert bad.status_code == 400
    escape = c.post(
        "/api/documents/upload",
        json={"filename": "a.md", "folder": "../..", "content_base64": content},
    )
    assert escape.status_code == 400
    assert (
        c.post(
            "/api/documents/upload", json={"filename": "a.md", "content_base64": "%%%"}
        ).status_code
        == 400
    )


def test_sources_carry_relevance_between_0_and_1(make_env: Any) -> None:
    env = make_env()
    c = _client(env)
    c.post("/ingest", json={})
    body = c.post("/api/query", json={"question": Q, "retrieve_only": True}).json()
    rel = [s["relevance"] for s in body["sources"]]
    assert rel and all(0.0 <= x <= 1.0 for x in rel)
    assert rel == sorted(rel, reverse=True)
