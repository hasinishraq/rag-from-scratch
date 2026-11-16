"""Connection test: the same five stages for every provider, stopping at the first failure.
(1) settings complete (2) server reachable (3) model available (4) generates a short reply
(5) ready for RAG. Goes through the gateway, bypasses the cache, counts against quota."""

from __future__ import annotations

import logging
import time
from datetime import datetime

from rag.config import HealthcheckConfig
from rag.errors import (
    KeyMissing,
    ModelNotFound,
    ProviderAuthError,
    ProviderError,
    ProviderNotFree,
    ProviderUnreachable,
    QuotaExhausted,
    RateLimited,
    RemoteOllamaNotConfirmed,
)
from rag.generate import citations
from rag.generate import prompts as T
from rag.llm.gateway import Gateway
from rag.llm.ollama import OllamaClient, same_model
from rag.llm.settings import SettingsStore
from rag.types import (
    ChatMessage,
    ChatRequest,
    Provider,
    ProviderKind,
    StageResult,
    TestResult,
    TestStatus,
)

logger = logging.getLogger(__name__)
STAGES = (
    "Settings complete",
    "Server reachable",
    "Model available",
    "Generates a reply",
    "Ready for RAG",
)
_PROBE_MAX_TOKENS = 200


class _Stop(Exception):
    def __init__(self, status: TestStatus, detail: str, retry_after: float | None = None):
        super().__init__(detail)
        self.status, self.detail, self.retry_after = status, detail, retry_after


def _rate_text(retry_after_s: float | None) -> str:
    """Plain message for the provider card; the provider's raw error goes to the log."""
    when = f"in {int(retry_after_s) + 1} s" if retry_after_s else "in a minute"
    return f"The provider is busy or this key reached its limit. Test again {when}."


def _status_for(exc: Exception) -> TestStatus:
    if isinstance(exc, (RateLimited, QuotaExhausted)):
        return TestStatus.RATE_LIMITED
    if isinstance(exc, KeyMissing):
        return TestStatus.KEY_MISSING
    if isinstance(exc, ProviderAuthError):
        return TestStatus.KEY_REJECTED
    if isinstance(exc, ModelNotFound):
        return TestStatus.MODEL_NOT_FOUND
    return TestStatus.UNREACHABLE


class ConnectionTester:
    def __init__(
        self,
        gateway: Gateway,
        settings: SettingsStore,
        ollama: OllamaClient,
        cfg: HealthcheckConfig,
    ) -> None:
        self.gateway, self.settings, self.ollama, self.cfg = gateway, settings, ollama, cfg

    def test_all(self) -> list[TestResult]:
        # One after another, never in parallel, to stay inside per-minute limits.
        return [self.test(p.name) for p in self.settings.list() if p.enabled]

    def test(self, name: str) -> TestResult:
        p = self.settings.get(name)
        stages: list[StageResult] = []
        counters = {"requests": 0, "tokens": 0}
        latency: int | None = None
        status, detail, retry = TestStatus.WORKING, "All five stages passed.", None
        try:
            self._settings_stage(p, stages)
            listed = self._reachable_stage(p, stages)
            self._model_stage(p, listed, stages)
            latency = self._reply_stage(p, stages, counters)
            warning = self._ready_stage(p, stages, counters)
            if warning:
                status, detail = TestStatus.WORKING_WITH_WARNING, warning
        except _Stop as stop:
            status, detail, retry = stop.status, stop.detail, stop.retry_after
            stages.append(
                StageResult(
                    n=len(stages) + 1,
                    name=STAGES[len(stages)],
                    outcome="failed",
                    detail=stop.detail,
                )
            )
        result = TestResult(
            provider=p.name,
            status=status,
            stages=tuple(stages),
            latency_ms=latency,
            detail=detail,
            requests_used=counters["requests"],
            tokens_used=counters["tokens"],
            tested_at=datetime.now().isoformat(timespec="seconds"),
            retry_after_s=retry,
        )
        self.settings.record_test(result)
        return result

    def _timeout(self, p: Provider) -> float:
        local = p.kind is ProviderKind.OLLAMA
        return self.cfg.local_timeout_s if local else self.cfg.hosted_timeout_s

    # ---- stage 1 ----
    def _settings_stage(self, p: Provider, stages: list[StageResult]) -> None:
        if not p.base_url:
            raise _Stop(TestStatus.UNREACHABLE, "No address set.")
        if not p.model:
            what = (
                "Select an installed model."
                if p.kind is ProviderKind.OLLAMA
                else "Enter a model id."
            )
            raise _Stop(TestStatus.MODEL_NOT_FOUND, f"No model selected. {what}")
        try:
            self.gateway.preflight(p)  # same checks the gateway applies before sending
        except KeyMissing:
            raise _Stop(TestStatus.KEY_MISSING, "No API key saved. Paste a key and save.") from None
        except (ProviderNotFree, RemoteOllamaNotConfirmed) as exc:
            raise _Stop(TestStatus.UNTESTED, exc.message) from None
        detail = (
            "Address set, model selected."
            if p.kind is ProviderKind.OLLAMA
            else "Address, model and key present. Marked as free."
        )
        stages.append(StageResult(n=1, name=STAGES[0], outcome="passed", detail=detail))

    # ---- stage 2 ----
    def _reachable_stage(self, p: Provider, stages: list[StageResult]) -> list[str] | None:
        try:
            if p.kind is ProviderKind.OLLAMA:
                listed = [m.name for m in self.ollama.list_models(p.base_url, self._timeout(p))]
                detail = f"Ollama answered at {p.base_url}."
            else:
                listed = self.gateway.list_models(p, self._timeout(p))
                detail = "Key accepted." if listed is not None else "Server answered."
        except ProviderUnreachable as exc:
            msg = (
                f"Ollama is not running at this address ({p.base_url}). Start Ollama, then "
                "test again."
                if p.kind is ProviderKind.OLLAMA
                else f"{exc.message}. Check the address and your connection."
            )
            raise _Stop(TestStatus.UNREACHABLE, msg) from None
        except ProviderAuthError:
            raise _Stop(
                TestStatus.KEY_REJECTED,
                "The provider refused the key. Create a new one in its console.",
            ) from None
        except RateLimited as exc:
            raise _Stop(
                TestStatus.RATE_LIMITED, _rate_text(exc.retry_after_s), exc.retry_after_s
            ) from None
        except ProviderError as exc:
            raise _Stop(TestStatus.UNREACHABLE, exc.message) from None
        stages.append(StageResult(n=2, name=STAGES[1], outcome="passed", detail=detail))
        return listed

    # ---- stage 3 ----
    def _model_stage(
        self, p: Provider, listed: list[str] | None, stages: list[StageResult]
    ) -> None:
        if listed is None:
            stages.append(
                StageResult(
                    n=3,
                    name=STAGES[2],
                    outcome="not_checked",
                    detail="The provider has no model-list endpoint.",
                )
            )
            return
        if p.kind is ProviderKind.OLLAMA:
            found = any(same_model(p.model, m) for m in listed)
            hint = f"Run: ollama pull {p.model}"
        else:
            found = p.model.removeprefix("models/") in listed
            hint = "Load the model list and pick an id from it."
        if not found:
            raise _Stop(TestStatus.MODEL_NOT_FOUND, f"{p.model} is not offered. {hint}")
        stages.append(
            StageResult(n=3, name=STAGES[2], outcome="passed", detail=f"{p.model} is available.")
        )

    # ---- stages 4 and 5 ----
    def _ask(
        self,
        p: Provider,
        messages: tuple[ChatMessage, ...],
        max_tokens: int,
        counters: dict[str, int],
    ) -> tuple[str, int]:
        req = ChatRequest(
            messages=messages, temperature=0.0, max_tokens=max_tokens, purpose="healthcheck"
        )
        counters["requests"] += 1
        try:
            resp = self.gateway.call_provider(p, req, self._timeout(p), retry=False)
        except (RateLimited, QuotaExhausted) as exc:
            retry = exc.retry_after_s if isinstance(exc, RateLimited) else None
            logger.info("rate-limited during test of %s: %s", p.name, exc.message)
            raise _Stop(TestStatus.RATE_LIMITED, _rate_text(retry), retry) from None
        except ProviderError as exc:
            raise _Stop(_status_for(exc), exc.message) from None
        counters["tokens"] += resp.tokens_in + resp.tokens_out
        return resp.text, resp.latency_ms

    def _reply_stage(self, p: Provider, stages: list[StageResult], counters: dict[str, int]) -> int:
        msgs = (ChatMessage(role="user", content=T.REPLY_OK_PROMPT),)
        text, first = self._ask(p, msgs, self.cfg.reply_max_tokens, counters)
        latency, detail = first, f"Replied in {first / 1000:.1f} s."
        if p.kind is ProviderKind.OLLAMA:  # first call can include model load time
            started = time.perf_counter()
            self._ask(p, msgs, self.cfg.reply_max_tokens, counters)
            latency = int((time.perf_counter() - started) * 1000)
            detail = (
                f"First call {first / 1000:.1f} s including model load, "
                f"then {latency / 1000:.1f} s."
            )
        if not text:
            detail += " The reply was empty (reasoning models may need more output tokens)."
        stages.append(StageResult(n=4, name=STAGES[3], outcome="passed", detail=detail))
        return latency

    def _ready_stage(self, p: Provider, stages: list[StageResult], counters: dict[str, int]) -> str:
        answerable = T.build_messages(T.PROBE_ANSWERABLE, [T.PROBE_SOURCE])
        unanswerable = T.build_messages(T.PROBE_UNANSWERABLE, [T.PROBE_SOURCE])
        a, _ = self._ask(p, answerable, _PROBE_MAX_TOKENS, counters)
        u, _ = self._ask(p, unanswerable, _PROBE_MAX_TOKENS, counters)
        cites = 1 in citations.markers(a)
        refuses = citations.is_refusal(u)
        notes = [
            "Cited the source correctly." if cites else "Did not cite the source with [1].",
            "Refused the unanswerable question."
            if refuses
            else "Answered the unanswerable question when it should have refused.",
        ]
        ok = cites and refuses
        stages.append(
            StageResult(
                n=5, name=STAGES[4], outcome="passed" if ok else "warning", detail=" ".join(notes)
            )
        )
        return (
            ""
            if ok
            else "Answers, but " + ("weak at citing" if not cites else "weak at refusing") + "."
        )
