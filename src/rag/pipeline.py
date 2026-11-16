"""answer(question) -> Answer, the one entry point used by the API, the CLI and the evaluation
runner. Every call writes one trace row."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from rag.errors import InvalidRequest, RagError
from rag.generate import citations
from rag.generate.context import build_context
from rag.generate.guard import suspicious_sources
from rag.generate.prompts import build_messages, render_prompt
from rag.ingest.runner import Progress, ingest
from rag.retrieve.hybrid import HybridRetriever
from rag.retrieve.rerank import rerank
from rag.services import Services
from rag.types import (
    Answer,
    Candidate,
    ChatRequest,
    ChatResponse,
    Filters,
    IngestReport,
    Mode,
    SourceRef,
    StreamEvent,
    Trace,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")
# Receives progress while an answer is prepared. Raising Cancelled from it stops the answer.
OnEvent = Callable[[StreamEvent], None]


def _relevance(c: Candidate, best_rrf: float) -> float:
    if c.rerank_score is not None:
        return max(0.0, min(1.0, c.rerank_score))
    return c.rrf_score / best_rrf if best_rrf > 0 else 0.0


def _refs(sources: Sequence[Candidate], cited: Sequence[int] = ()) -> tuple[SourceRef, ...]:
    best_rrf = max((c.rrf_score for c in sources), default=0.0)
    return tuple(
        SourceRef(
            n=i,
            chunk_id=c.chunk_id,
            title=c.title,
            section_path=c.section_path,
            page=c.page,
            path=c.path,
            text=c.text,
            score=c.rerank_score if c.rerank_score is not None else c.rrf_score,
            cited=i in cited,
            relevance=_relevance(c, best_rrf),
        )
        for i, c in enumerate(sources, start=1)
    )


class Pipeline:
    def __init__(self, services: Services) -> None:
        self.s = services
        self.cfg = services.cfg
        r = self.cfg.retrieval
        self.retriever = HybridRetriever(services.indexes, r.k_retrieve, r.rrf_k, r.use_bm25)

    def ingest(
        self, path: str | None = None, rebuild: bool = False, progress: Progress | None = None
    ) -> IngestReport:
        if progress is None:
            return ingest(self.cfg, self.s.indexes, self.s.tok, path, rebuild)
        return ingest(self.cfg, self.s.indexes, self.s.tok, path, rebuild, progress)

    def _rank(
        self, question: str, filters: Filters | None, keep: int, run: _Run
    ) -> tuple[list[Candidate], list[Candidate]]:
        """Returns (all fused candidates, the best `keep` after reranking)."""
        fused = run.timed("retrieve_ms", lambda: self.retriever.retrieve(question, filters))
        reranker = self.s.reranker
        if reranker is None:
            return fused, fused[:keep]
        k = self.cfg.rerank.k_rerank
        # Keep every reranker score for the trace, not only the passages sent to the model.
        ranked = run.timed("rerank_ms", lambda: rerank(reranker, question, fused, k, k))
        scored = {c.chunk_id: c for c in ranked}
        return [scored.get(c.chunk_id, c) for c in fused], ranked[:keep]

    def warm_up(self) -> None:
        """Loads the local models and runs them once on passage-sized input, so the first real
        question does not pay for model loading or the CPU kernels' first-run set-up."""
        passage = "warm up " * (self.cfg.chunking.chunk_tokens // 2)
        self.s.indexes.embedder.embed_query("warm up")
        if self.s.reranker is not None:
            self.s.reranker.score("warm up question", [passage] * self.cfg.rerank.k_context)

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        filters: Filters | None = None,
        mode: Mode | None = None,
        retrieve_only: bool = False,
        on_event: OnEvent | None = None,
    ) -> Answer:
        question = question.strip()
        if not question:
            raise InvalidRequest("question is empty")
        if len(question) > self.cfg.limits.max_question_chars:
            raise InvalidRequest(
                f"question is longer than {self.cfg.limits.max_question_chars} characters"
            )
        run = _Run(started=time.perf_counter())
        mode = mode or self.s.settings.mode()
        keep = top_k or self.cfg.rerank.k_context
        candidates, top = self._rank(question, filters, keep, run)
        gen = self.cfg.generation
        ctx = build_context(
            top,
            self.s.tok,
            gen.max_context_tokens,
            keep,
            min_score=self.cfg.rerank.min_source_score if self.s.reranker else 0.0,
            trim=gen.trim_overlap,
        )
        sources = ctx.sources
        run.trace = Trace(
            trace_id=uuid.uuid4().hex[:16],
            question=question,
            candidates=tuple(candidates),
            prompt="",
            answer="",
            refused=False,
            provider=None,
            model=None,
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
            cached=False,
            config_hash=self.cfg.hash(),
            mode=mode.value,
        )
        flagged = suspicious_sources(ctx.prompt_texts)
        run.warnings += tuple(
            f"source [{n}] contains text that looks like instructions to the model; it was "
            "treated as data"
            for n in flagged
        )
        best = top[0].rerank_score if top else None
        below = self.s.reranker is not None and (best is None or best < self.cfg.rerank.min_score)
        if retrieve_only:
            return self._finish(run, "", False, _refs(sources), None)
        if not sources or below:
            # Refuse without calling the generator: saves quota, avoids answers on weak context.
            return self._finish(run, gen.refusal_text, True, _refs(sources), None)
        # Savings are counted only when a prompt is really sent: a passages-only lookup or a
        # refusal before generation sends nothing, so nothing was saved.
        run.tokens_saved = ctx.tokens_saved
        messages = build_messages(
            question,
            [(c.title, c.section_path, t) for c, t in zip(sources, ctx.prompt_texts, strict=True)],
        )
        run.trace = run.trace.model_copy(update={"prompt": render_prompt(messages)})
        request = ChatRequest(
            messages=messages, temperature=gen.temperature, max_tokens=gen.max_output_tokens
        )
        try:
            resp = run.timed(
                "generate_ms", lambda: self._generate(request, mode, sources, on_event)
            )
        except RagError as exc:
            run.errors += (f"{exc.code}: {exc.message}",)
            self._finish(run, "", False, (), None)
            raise
        resp = resp.model_copy(update={"text": citations.normalise_markers(resp.text)})
        if citations.is_refusal(resp.text):
            return self._finish(run, gen.refusal_text, True, _refs(sources), resp)
        check = citations.check(resp.text, len(sources))
        run.errors += tuple(f"unknown citation [{n}]" for n in check.unknown)
        run.warnings = run.errors + run.warnings
        if check.uncited_sentences:
            run.warnings += ("some sentences have no citation",)
        if not check.cited:
            run.warnings += ("the answer cites no source",)
        if resp.tokens_out >= gen.max_output_tokens:
            # Reasoning models spend part of the budget before the first visible word.
            run.warnings += (
                f"the answer reached the output limit ({gen.max_output_tokens} tokens) and may "
                "be cut short",
            )
        return self._finish(run, resp.text, False, _refs(sources, check.cited), resp)

    def _generate(
        self,
        request: ChatRequest,
        mode: Mode,
        sources: Sequence[Candidate],
        on_event: OnEvent | None,
    ) -> ChatResponse:
        """The model call. With a listener, the passages are reported first and the answer is
        streamed; the result is the same ChatResponse either way."""
        if on_event is None:
            return self.s.gateway.chat(request, mode)
        on_event(StreamEvent(kind="sources", sources=_refs(sources)))
        gate = citations.RefusalGate(lambda text: on_event(StreamEvent(kind="delta", text=text)))
        return self.s.gateway.chat(request, mode, on_delta=gate.feed)

    def _finish(
        self,
        run: _Run,
        text: str,
        refused: bool,
        sources: tuple[SourceRef, ...],
        resp: ChatResponse | None,
    ) -> Answer:
        latency = int((time.perf_counter() - run.started) * 1000)
        timings = {**run.timings, "total_ms": latency}
        if run.trace is None:  # set before any _finish call; guards a programming error
            raise RuntimeError("trace not initialised")
        trace = run.trace.model_copy(
            update={
                "answer": text,
                "refused": refused,
                "latency_ms": latency,
                "errors": run.errors,
                "provider": resp.provider if resp else None,
                "model": resp.model if resp else None,
                "tokens_in": resp.tokens_in if resp else 0,
                "tokens_out": resp.tokens_out if resp else 0,
                "cached": resp.cached if resp else False,
                "timings": timings,
                "tokens_saved": run.tokens_saved,
            }
        )
        self.s.chunks.write_trace(trace)
        return Answer(
            answer=text,
            refused=refused,
            sources=sources,
            trace_id=trace.trace_id,
            provider=trace.provider,
            model=trace.model,
            cached=trace.cached,
            latency_ms=latency,
            n_candidates=len(trace.candidates),
            warnings=run.warnings,
            timings=timings,
            tokens_in=trace.tokens_in,
            tokens_out=trace.tokens_out,
            tokens_saved=run.tokens_saved,
        )


@dataclass
class _Run:
    """What one answer() call accumulates before its trace is written."""

    started: float
    trace: Trace | None = None
    timings: dict[str, int] = field(default_factory=dict)
    tokens_saved: int = 0
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def timed(self, name: str, fn: Callable[[], T]) -> T:
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            self.timings[name] = self.timings.get(name, 0) + int((time.perf_counter() - t0) * 1000)
