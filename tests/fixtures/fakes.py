"""Deterministic fakes: embedder, reranker, clock, and a fake model server (OpenAI-compatible hosted
providers plus Ollama) served through httpx.MockTransport, so tests make no network request."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import httpx
import numpy as np
import numpy.typing as npt

from rag.index.bm25 import tokenize
from rag.index.embedder import normalise

DIM = 64


def _hash_vec(text: str) -> npt.NDArray[np.float32]:
    vec = np.zeros(DIM, dtype=np.float32)
    for term in tokenize(text):
        h = int(hashlib.md5(term.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0 if (h >> 8) % 2 else -1.0
    return vec


class FakeEmbedder:
    model_id = "fake:hash"

    def embed_passages(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        return normalise(np.stack([_hash_vec(t) for t in texts]))

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        return normalise(_hash_vec(text))[0]


class FakeReranker:
    """Share of the question's content words that appear in the passage."""

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        q = set(tokenize(question))
        return [len(q & set(tokenize(t))) / max(1, len(q)) for t in texts]


class FakeClock:
    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


_SOURCE = re.compile(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', re.S)


def fake_answer(prompt: str, refuses: bool = True, cites: bool = True) -> str:
    """Answers from the best-overlapping source with a citation, or NOT_IN_SOURCES."""
    if "Reply with the word OK" in prompt:
        return "OK"
    question = prompt.rsplit("Question:", 1)[-1]
    q_terms = set(tokenize(question))
    best, best_n, best_overlap = "", 0, 0
    for n, text in _SOURCE.findall(prompt):
        overlap = len(q_terms & set(tokenize(text)))
        if overlap > best_overlap:
            best, best_n, best_overlap = text, int(n), overlap
    if best_overlap < max(2, len(q_terms) // 2):
        return "NOT_IN_SOURCES" if refuses else "It is probably about 5,000 books."
    sentence = best.split(". ")[0].rstrip(".")
    return f"{sentence} [{best_n}]." if cites else f"{sentence}."


class _SseBody(httpx.SyncByteStream):
    """A streamed chat reply: one server-sent event per word, usage in the last one."""

    def __init__(self, text: str, usage: dict[str, int], breaks: bool) -> None:
        self.text, self.usage, self.breaks = text, usage, breaks

    def __iter__(self) -> Iterator[bytes]:
        words = re.findall(r"\S+\s*", self.text)
        for i, word in enumerate(words):
            chunk: dict[str, object] = {"choices": [{"delta": {"content": word}}]}
            if i == len(words) - 1:
                chunk["usage"] = self.usage
            yield f"data: {json.dumps(chunk)}\n\n".encode()
            if self.breaks:
                raise httpx.ReadError("connection reset")
        yield b"data: [DONE]\n\n"


@dataclass
class HostBehaviour:
    mode: str = "ok"  # ok | bad_key | rate_limited | daily_limit | server_error | unreachable
    models: list[str] = field(default_factory=list)
    list_endpoint: bool = True
    refuses: bool = True
    cites: bool = True
    retry_after: str = "2"
    fail_times: int = 10**6  # how many chat calls fail before the host recovers
    break_stream: bool = False  # a streamed reply drops after its first piece


@dataclass
class FakeModelServer:
    hosts: dict[str, HostBehaviour] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=self.transport())

    def chat_requests(self, host: str | None = None) -> list[httpx.Request]:
        return [
            r
            for r in self.requests
            if r.url.path.endswith("/chat/completions") and (host is None or r.url.host == host)
        ]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        b = self.hosts.get(host)
        if b is None or b.mode == "unreachable":
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if path.endswith("/api/tags"):
            models = [
                {"name": m, "size": 2_000_000_000, "details": {"family": "llama"}} for m in b.models
            ]
            return httpx.Response(200, json={"models": models})
        if path.endswith("/api/embed"):
            texts = json.loads(request.content)["input"]
            return httpx.Response(200, json={"embeddings": [_hash_vec(t).tolist() for t in texts]})
        is_ollama = path.startswith("/v1/") and request.url.port == 11434
        if not is_ollama and request.headers.get("authorization") != "Bearer good-key":
            return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})
        if b.mode == "bad_key":
            return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})
        if path.endswith("/models"):
            if not b.list_endpoint:
                return httpx.Response(404, text="not found")
            return httpx.Response(200, json={"data": [{"id": m} for m in b.models]})
        if path.endswith("/chat/completions"):
            return self._chat(request, b)
        return httpx.Response(404, text="not found")

    def _chat(self, request: httpx.Request, b: HostBehaviour) -> httpx.Response:
        body = json.loads(request.content)
        if b.mode != "ok" and b.fail_times > 0:
            b.fail_times -= 1
            if b.mode == "rate_limited":
                return httpx.Response(
                    429,
                    headers={"retry-after": b.retry_after},
                    json={"error": {"message": "Rate limit reached (RPM)"}},
                )
            if b.mode == "daily_limit":
                return httpx.Response(
                    429, json={"error": {"message": "Rate limit reached on tokens per day (TPD)"}}
                )
            if b.mode == "server_error":
                return httpx.Response(500, text="upstream error")
        if body["model"] not in b.models:
            return httpx.Response(404, json={"error": {"message": "The model does not exist"}})
        prompt = "\n".join(m["content"] for m in body["messages"])
        text = fake_answer(prompt, b.refuses, b.cites)
        usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": len(text.split())}
        if body.get("stream"):
            return httpx.Response(200, stream=_SseBody(text, usage, b.break_stream))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": text}}], "usage": usage},
        )
