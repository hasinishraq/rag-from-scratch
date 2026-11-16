"""Data that crosses a module boundary."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


# ---------- documents and chunks ----------


class Block(Frozen):
    """One heading-scoped paragraph of a parsed document."""

    text: str
    section_path: tuple[str, ...] = ()
    page: int | None = None


class Document(Frozen):
    doc_id: str
    path: str
    title: str
    blocks: tuple[Block, ...]
    sha256: str = ""


class Chunk(Frozen):
    chunk_id: str
    doc_id: str
    ord: int
    text: str
    embed_text: str
    section_path: str
    page: int | None
    n_tokens: int


class StoredChunk(Frozen):
    chunk_id: str
    doc_id: str
    ord: int
    text: str
    section_path: str
    page: int | None
    n_tokens: int
    title: str
    path: str


class DocumentRecord(Frozen):
    doc_id: str
    path: str
    sha256: str
    title: str
    n_chunks: int
    ingested_at: str


class IngestReport(Frozen):
    added: int
    updated: int
    deleted: int
    chunks: int
    errors: tuple[str, ...] = ()


# ---------- retrieval ----------


class Filters(Frozen):
    """Restricts retrieval to documents whose relative path starts with one of these prefixes."""

    path_prefixes: tuple[str, ...] = ()


class Candidate(Frozen):
    chunk_id: str
    doc_id: str
    text: str
    title: str
    section_path: str
    page: int | None
    path: str
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


class SourceRef(Frozen):
    n: int
    chunk_id: str
    title: str
    section_path: str
    page: int | None
    path: str
    text: str
    score: float | None
    cited: bool = False
    # 0..1 for display: the reranker score, or the RRF score relative to the best candidate
    # when the reranker is off (raw RRF values are tiny and not comparable across queries).
    relevance: float = 0.0


# ---------- model calls ----------


class ChatMessage(Frozen):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(Frozen):
    messages: tuple[ChatMessage, ...]
    temperature: float = 0.1
    max_tokens: int = 400
    purpose: str = "answer"


class ChatResponse(Frozen):
    text: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cached: bool = False
    attempts: tuple[str, ...] = ()


class ProviderKind(StrEnum):
    OPENAI_COMPATIBLE = "openai_compatible"
    OLLAMA = "ollama"


class Mode(StrEnum):
    AUTO = "auto"
    FREE_API_ONLY = "free_api_only"
    OLLAMA_ONLY = "ollama_only"


class TestStatus(StrEnum):
    WORKING = "working"
    WORKING_WITH_WARNING = "working_with_warning"
    RATE_LIMITED = "rate_limited"
    KEY_MISSING = "key_missing"
    KEY_REJECTED = "key_rejected"
    UNREACHABLE = "unreachable"
    MODEL_NOT_FOUND = "model_not_found"
    UNTESTED = "untested"


USABLE_IN_AUTO = frozenset(
    {TestStatus.WORKING, TestStatus.WORKING_WITH_WARNING, TestStatus.RATE_LIMITED}
)


class Limits(Frozen):
    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    tpd: int | None = None


class Provider(Frozen):
    """Provider settings as stored. Never holds the key itself."""

    name: str
    kind: ProviderKind
    base_url: str
    model: str
    api_key_env: str = ""
    limits: Limits = Limits()
    free: bool = False
    enabled: bool = True
    position: int = 0
    allow_remote: bool = False
    test_status: TestStatus = TestStatus.UNTESTED
    test_stage: int | None = None
    test_detail: str = ""
    test_latency_ms: int | None = None
    tested_at: str | None = None


class StageResult(Frozen):
    n: int
    name: str
    outcome: Literal["passed", "failed", "warning", "not_checked", "skipped"]
    detail: str


class TestResult(Frozen):
    __test__ = False  # not a pytest class

    provider: str
    status: TestStatus
    stages: tuple[StageResult, ...]
    latency_ms: int | None
    detail: str
    requests_used: int
    tokens_used: int
    tested_at: str
    retry_after_s: float | None = None


class OllamaModel(Frozen):
    name: str
    size_bytes: int
    kind: Literal["chat", "embeddings"]


# ---------- answers and traces ----------


class Answer(Frozen):
    answer: str
    refused: bool
    sources: tuple[SourceRef, ...]
    trace_id: str
    provider: str | None
    model: str | None
    cached: bool
    latency_ms: int
    n_candidates: int
    warnings: tuple[str, ...] = ()
    timings: dict[str, int] = Field(default_factory=dict)  # retrieve_ms, rerank_ms, generate_ms
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_saved: int = 0  # by dropping weak passages and repeated overlap


class StreamEvent(Frozen):
    """What answer() reports while it works, for clients that show the answer as it is written:
    the passages once they are chosen, then each piece of text."""

    kind: Literal["sources", "delta"]
    sources: tuple[SourceRef, ...] = ()
    text: str = ""


class Trace(Frozen):
    trace_id: str
    question: str
    candidates: tuple[Candidate, ...]
    prompt: str
    answer: str
    refused: bool
    provider: str | None
    model: str | None
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cached: bool
    config_hash: str
    mode: str
    errors: tuple[str, ...] = ()
    timings: dict[str, int] = Field(default_factory=dict)
    tokens_saved: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    feedback: Literal["up", "down"] | None = None  # the reader's verdict on the answer
    feedback_note: str = ""
