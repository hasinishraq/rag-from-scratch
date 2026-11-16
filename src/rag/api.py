"""HTTP interface (ENG-RAG-001 §6.9). Errors are JSON {code, message}; quota exhaustion is 503."""

from __future__ import annotations

import base64
import binascii
import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from rag import eval_view
from rag.config import load_config
from rag.errors import Busy, Conflict, InvalidRequest, NotFound, RagError
from rag.index.chunk_store import TRACE_STATUSES
from rag.ingest.jobs import IngestJobs
from rag.ingest.runner import remove_file, save_upload
from rag.llm.gateway import ROUTING_STRATEGY
from rag.llm.providers import is_local_url
from rag.llm.settings import EMBEDDING_BACKEND, ProviderInput, ProviderUpdate
from rag.observability import Metrics, configure_logging, make_request_context, request_id_var
from rag.ops import app_version, ops_router
from rag.pipeline import OnEvent, Pipeline
from rag.security import allowed_hosts, install_log_redaction, make_guard, redact
from rag.services import EMBEDDING_OLLAMA_MODEL, Services, build_services, make_embedder
from rag.streaming import SSE_HEADERS, answer_events
from rag.types import Answer, Filters, Mode, Provider, ProviderKind

logger = logging.getLogger(__name__)
PRESETS_FILE = Path("configs/provider_presets.yaml")
WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
# Base64 makes an upload a third larger than the file; the rest is room for the JSON around it.
_BASE64_GROWTH = 4 / 3
_BODY_SLACK_BYTES = 1024 * 1024
_MAX_TRACE_PAGE = 500
_TRACE_DETAIL = {"candidates", "prompt"}  # left out of trace lists asked for with summary=true


class QueryBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=20000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    filters: Filters | None = None
    mode: Mode | None = None
    retrieve_only: bool = False


class IngestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str | None = None
    rebuild: bool = False


class UploadBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=1, max_length=255)
    folder: str = ""
    content_base64: str


class FeedbackBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rating: Literal["up", "down"] | None  # null clears the verdict
    note: str = Field(default="", max_length=2000)


class ModeBody(BaseModel):
    mode: Mode


class RoutingBody(BaseModel):
    strategy: str = Field(pattern="^(ordered|fastest)$")


class EmbeddingBody(BaseModel):
    backend: str = Field(pattern="^(local|ollama)$")
    ollama_model: str = ""


def provider_view(s: Services, p: Provider) -> dict[str, Any]:
    """Public view of a provider: settings, key_set, quota and last test. Never the key."""
    data = p.model_dump(mode="json")
    data["key_set"] = p.kind is ProviderKind.OLLAMA or s.settings.key_set(p)
    data["quota"] = s.limiter.remaining(p.name, p.limits)
    data["local"] = p.kind is ProviderKind.OLLAMA and is_local_url(p.base_url)
    data["health"] = s.gateway.router.snapshot().get(p.name)
    return data


def _error(code: str, message: str, status: int) -> JSONResponse:
    body = {"code": code, "message": message, "request_id": request_id_var.get()}
    return JSONResponse(body, status_code=status)


def create_app(services: Services | None = None) -> FastAPI:
    install_log_redaction()
    configure_logging()
    metrics = Metrics()
    # Services passed in (tests) belong to the caller; ones built here are closed at shutdown.
    state: dict[str, Any] = {"services": services, "pipeline": None, "owned": services is None}
    preload = os.environ.get("RAG_PRELOAD", "1") != "0"
    state["warm"] = not preload
    # Re-entrant: pipe() builds the services through svc() while holding this lock.
    lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Load the local models in the background so the first question is not the slow one.
        if preload:
            threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()
        yield
        if state["owned"] and state["services"] is not None:
            state["services"].close()
            logger.info("stopped: database and HTTP client closed")

    def _warm_up() -> None:
        try:
            _prune_traces()
            pipe().warm_up()
            logger.info("local models loaded")
        except Exception as exc:  # warm-up is an optimisation; the first query retries
            logger.warning("warm-up failed: %s", type(exc).__name__)
        finally:
            state["warm"] = True

    def _prune_traces() -> None:
        s = svc()
        cutoff = datetime.now() - timedelta(days=s.cfg.tracing.retention_days)
        removed = s.chunks.prune_traces(
            cutoff.isoformat(timespec="seconds"), s.cfg.tracing.max_rows
        )
        if removed:
            logger.info("trace retention: removed %d old traces", removed)

    def _max_body_bytes() -> int:
        file_bytes = svc().cfg.limits.max_file_mb * 1024 * 1024
        return int(file_bytes * _BASE64_GROWTH) + _BODY_SLACK_BYTES

    app = FastAPI(title="RAG from scratch", version=app_version(), lifespan=lifespan)
    app.middleware("http")(make_guard(allowed_hosts()))
    # Registered last, so it runs first: refused requests get an id and are counted too.
    app.middleware("http")(make_request_context(metrics, _max_body_bytes))
    r = APIRouter()

    def svc() -> Services:
        with lock:
            if state["services"] is None:
                cfg = load_config(os.environ.get("RAG_CONFIG", "configs/default.yaml"))
                state["services"] = build_services(cfg)
            return state["services"]

    def jobs() -> IngestJobs:
        with lock:
            if state.get("jobs") is None:
                state["jobs"] = IngestJobs(
                    lambda path, rebuild, progress: pipe().ingest(path, rebuild, progress)
                )
            return state["jobs"]

    def pipe() -> Pipeline:
        with lock:
            if state["pipeline"] is None:
                state["pipeline"] = Pipeline(state["services"] or svc())
            return state["pipeline"]

    def query_slots() -> threading.BoundedSemaphore:
        with lock:
            if state.get("slots") is None:
                n = svc().cfg.limits.max_concurrent_queries
                state["slots"] = threading.BoundedSemaphore(n)
            return state["slots"]

    def run_query(body: QueryBody, on_event: OnEvent | None = None) -> Answer:
        return pipe().answer(
            body.question, body.top_k, body.filters, body.mode, body.retrieve_only, on_event
        )

    def take_slot() -> threading.BoundedSemaphore:
        slots = query_slots()
        if not slots.acquire(blocking=False):
            raise Busy("too many questions are being answered at once; try again in a moment")
        return slots

    @app.exception_handler(RagError)
    async def _rag_error(_: Request, exc: RagError) -> JSONResponse:
        return _error(exc.code, redact(exc.message), exc.http_status)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return _error("invalid_request", msg, 422)

    @app.exception_handler(Exception)
    async def _internal(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error: %s", type(exc).__name__)
        return _error("internal_error", "internal error", 500)

    # ---------- queries and documents ----------
    @r.post("/query")
    def query(body: QueryBody) -> dict[str, Any]:
        slots = take_slot()
        try:
            return run_query(body).model_dump(mode="json")
        finally:
            slots.release()

    @r.post("/query/stream")
    def query_stream(body: QueryBody) -> StreamingResponse:
        """The same answer as POST /query, as server-sent events: `sources`, `delta` for each
        piece of text, then `done` with the complete answer, or `error` with {code, message}."""
        slots = take_slot()

        def run(on_event: OnEvent) -> Answer:
            try:
                return run_query(body, on_event)
            finally:
                slots.release()

        return StreamingResponse(
            answer_events(run), media_type="text/event-stream", headers=SSE_HEADERS
        )

    @r.post("/ingest")
    def ingest_route(body: IngestBody | None = None) -> dict[str, Any]:
        body = body or IngestBody()
        return pipe().ingest(body.path, body.rebuild).model_dump(mode="json")

    @r.post("/ingest/jobs", status_code=202)
    def start_ingest(body: IngestBody | None = None) -> dict[str, Any]:
        """Starts indexing in the background; poll GET /ingest/jobs/current for progress."""
        body = body or IngestBody()
        return jobs().start(body.path, body.rebuild)

    @r.get("/ingest/jobs/current")
    def ingest_status() -> dict[str, Any]:
        return {"job": jobs().status()}

    @r.get("/sources/{chunk_id}")
    def source(chunk_id: str) -> dict[str, Any]:
        chunk = svc().chunks.get(chunk_id)
        if chunk is None:
            raise NotFound(f"chunk not found: {chunk_id}")
        return {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "title": chunk.title,
            "section_path": chunk.section_path,
            "page": chunk.page,
            "path": chunk.path,
        }

    @r.get("/documents")
    def documents() -> dict[str, Any]:
        s = svc()
        docs = [d.model_dump() for d in s.chunks.documents()]
        return {
            "corpus_dir": str(s.cfg.corpus_dir),
            "documents": docs,
            "chunks": s.chunks.count(),
            "indexed_with": s.indexes.indexed_with(),
        }

    @r.post("/documents/upload", status_code=201)
    def upload(body: UploadBody) -> dict[str, Any]:
        """JSON upload (base64) so the browser UI can add files without a multipart parser."""
        s = svc()
        try:
            data = base64.b64decode(body.content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidRequest("content_base64 is not valid base64") from exc
        max_bytes = int(s.cfg.limits.max_file_mb * 1024 * 1024)
        saved = save_upload(s.cfg.corpus_dir, body.folder, body.filename, data, max_bytes)
        return {"saved": saved, "bytes": len(data)}

    @r.delete("/documents/{doc_id}")
    def delete_document(doc_id: str) -> dict[str, Any]:
        """Deletes the file from the corpus folder and its passages from all three stores."""
        job = jobs().status()
        if job is not None and job["status"] == "running":
            raise Conflict("indexing is running; remove the document when it has finished")
        s = svc()
        return {"deleted": remove_file(s.cfg.corpus_dir, s.indexes, doc_id)}

    @r.get("/traces")
    def traces(
        limit: int = 50, offset: int = 0, q: str = "", status: str = "", summary: bool = False
    ) -> JSONResponse:
        """Newest first. `q` searches questions and answers; `status` is answered, refused,
        lookup, failed or rated. `summary` leaves out the candidates and the prompt (the heavy
        parts) for lists. The X-Total-Count header carries the number of matches."""
        if status and status not in TRACE_STATUSES:
            raise InvalidRequest(f"status must be one of: {', '.join(sorted(TRACE_STATUSES))}")
        store = svc().chunks
        limit = max(1, min(limit, _MAX_TRACE_PAGE))
        rows = store.list_traces(limit, max(0, offset), q, status)
        return JSONResponse(
            [t.model_dump(mode="json", exclude=_TRACE_DETAIL if summary else None) for t in rows],
            headers={"X-Total-Count": str(store.count_traces(q, status))},
        )

    @r.get("/traces/{trace_id}")
    def trace(trace_id: str) -> dict[str, Any]:
        t = svc().chunks.get_trace(trace_id)
        if t is None:
            raise NotFound(f"trace not found: {trace_id}")
        return t.model_dump(mode="json")

    @r.put("/traces/{trace_id}/feedback")
    def feedback(trace_id: str, body: FeedbackBody) -> dict[str, Any]:
        if not svc().chunks.set_feedback(trace_id, body.rating, body.note.strip()):
            raise NotFound(f"trace not found: {trace_id}")
        return {"trace_id": trace_id, "rating": body.rating}

    # ---------- providers ----------
    @r.get("/providers")
    def list_providers() -> list[dict[str, Any]]:
        s = svc()
        return [provider_view(s, p) for p in s.settings.list()]

    @r.get("/providers/presets")
    def presets() -> list[dict[str, Any]]:
        if not PRESETS_FILE.exists():
            return []
        return (yaml.safe_load(PRESETS_FILE.read_text(encoding="utf-8")) or {}).get("presets", [])

    @r.post("/providers", status_code=201)
    def create_provider(body: ProviderInput) -> dict[str, Any]:
        s = svc()
        return provider_view(s, s.settings.create(body))

    @r.post("/providers/test-all")
    def test_all() -> list[dict[str, Any]]:
        return [r.model_dump(mode="json") for r in svc().tester.test_all()]

    @r.put("/providers/{name}")
    def update_provider(name: str, body: ProviderUpdate) -> dict[str, Any]:
        s = svc()
        return provider_view(s, s.settings.update(name, body))

    @r.delete("/providers/{name}")
    def delete_provider(name: str) -> dict[str, Any]:
        svc().settings.delete(name)
        return {"deleted": name}

    @r.post("/providers/{name}/test")
    def test_provider(name: str) -> dict[str, Any]:
        return svc().tester.test(name).model_dump(mode="json")

    @r.get("/providers/{name}/models")
    def provider_models(name: str) -> dict[str, Any]:
        s = svc()
        p = s.settings.get(name)
        if p.kind is ProviderKind.OLLAMA:
            running, models = s.ollama.status(p.base_url, s.cfg.healthcheck.hosted_timeout_s)
            return {"models": [m.name for m in models], "running": running}
        ids = s.gateway.list_models(p, s.cfg.healthcheck.hosted_timeout_s)
        return {"models": ids or [], "listed": ids is not None}

    @r.get("/ollama/models")
    def ollama_models(base_url: str | None = None, allow_remote: bool = False) -> dict[str, Any]:
        s = svc()
        url = base_url or s.settings.get("ollama").base_url
        if not is_local_url(url) and not allow_remote:
            raise InvalidRequest("this address is not on this computer; confirm 'allow remote'")
        running, models = s.ollama.status(url, s.cfg.healthcheck.hosted_timeout_s)
        return {"running": running, "base_url": url, "models": [m.model_dump() for m in models]}

    # ---------- settings ----------
    @r.get("/settings")
    def get_settings() -> dict[str, Any]:
        s = svc()
        backend = s.settings.get_setting(EMBEDDING_BACKEND, s.cfg.embedding.backend)
        return {
            "mode": s.settings.mode().value,
            "routing_strategy": s.settings.get_setting(
                ROUTING_STRATEGY, s.cfg.llm.routing.strategy
            ),
            "embedding_backend": backend,
            "embedding_model": s.indexes.embedder.model_id,
            "indexed_with": s.indexes.indexed_with(),
            "reranker": s.cfg.rerank.model if s.cfg.rerank.enabled else None,
            "test_cost": "3 requests and about 400 tokens per provider",
        }

    @r.put("/settings/routing")
    def set_routing(body: RoutingBody) -> dict[str, str]:
        svc().settings.set_setting(ROUTING_STRATEGY, body.strategy)
        return {"strategy": body.strategy}

    @r.get("/stats")
    def stats() -> dict[str, Any]:
        """Today's usage: volume, cache hits, tokens used and saved, latency, provider health."""
        s = svc()
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today = s.chunks.trace_stats(midnight.isoformat(timespec="seconds"))
        quota = {p.name: s.limiter.remaining(p.name, p.limits) for p in s.settings.list()}
        return {"today": today, "providers": s.gateway.router.snapshot(), "quota": quota}

    @r.put("/settings/mode")
    def set_mode(body: ModeBody) -> dict[str, str]:
        return {"mode": svc().settings.set_mode(body.mode).value}

    @r.put("/settings/embedding")
    def set_embedding(body: EmbeddingBody) -> dict[str, Any]:
        """Switching the embedding backend forces a full re-index (vectors are not comparable)."""
        s = svc()
        s.settings.set_setting(EMBEDDING_BACKEND, body.backend)
        s.settings.set_setting(EMBEDDING_OLLAMA_MODEL, body.ollama_model)
        s.indexes.embedder = make_embedder(s.cfg, s.settings, s.gateway)
        report = pipe().ingest(rebuild=True)
        return {"embedding_model": s.indexes.embedder.model_id, "reindexed": report.model_dump()}

    @r.get("/eval/runs")
    def eval_runs() -> list[dict[str, Any]]:
        return eval_view.list_runs()

    @r.get("/eval/runs/{run_id}")
    def eval_run(run_id: str) -> dict[str, Any]:
        return eval_view.run_detail(run_id, svc().chunks)

    # The browser UI calls /api/...; the bare paths stay for scripts and the documented contract.
    ops = ops_router(svc, metrics, lambda: bool(state["warm"]))
    for router in (r, ops):
        app.include_router(router)
        app.include_router(router, prefix="/api", include_in_schema=False)
    if WEB_DIST.is_dir():  # built web UI (npm --prefix web run build), served last
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    return app


app = create_app()
