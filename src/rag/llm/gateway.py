"""The only path to a model provider. Order of operations per request:
cache lookup -> pick provider -> quota check (RPM, RPD, TPM, TPD) -> call with timeout ->
on 429 honour Retry-After, else backoff with jitter (max attempts) -> fall back to the next
provider -> record usage -> store in cache."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence

import httpx

from rag.config import LLMConfig
from rag.errors import (
    EmptyResponse,
    KeyMissing,
    NoProviderAvailable,
    ProviderError,
    ProviderNotFree,
    ProviderUnreachable,
    QuotaExhausted,
    RagError,
    RateLimited,
    RemoteOllamaNotConfirmed,
    StreamInterrupted,
    TransientProviderError,
)
from rag.llm import providers as P
from rag.llm.cache import ResponseCache
from rag.llm.ratelimit import Clock, RateLimiter
from rag.llm.routing import Router
from rag.llm.secrets import SecretStore
from rag.llm.settings import SettingsStore
from rag.tokens import Tokenizer
from rag.types import (
    USABLE_IN_AUTO,
    ChatRequest,
    ChatResponse,
    Mode,
    Provider,
    ProviderKind,
    TestStatus,
)

logger = logging.getLogger(__name__)

ROUTING_STRATEGY = "routing_strategy"
# Receives each piece of a streamed answer as it arrives.
OnDelta = Callable[[str], None]
_HARD_FAILURES = frozenset(
    {
        TestStatus.KEY_MISSING,
        TestStatus.KEY_REJECTED,
        TestStatus.UNREACHABLE,
        TestStatus.MODEL_NOT_FOUND,
    }
)


class Gateway:
    def __init__(
        self,
        settings: SettingsStore,
        secrets: SecretStore,
        limiter: RateLimiter,
        cache: ResponseCache,
        tok: Tokenizer,
        cfg: LLMConfig,
        client: httpx.Client | None = None,
        clock: Clock | None = None,
        jitter: Callable[[], float] = random.random,
        router: Router | None = None,
    ) -> None:
        self.settings = settings
        self.secrets = secrets
        self.limiter = limiter
        self.cache = cache
        self.tok = tok
        self.cfg = cfg
        self.client = client or httpx.Client()
        self.clock = clock or limiter.clock
        self._jitter = jitter
        self.router = router or Router(cfg.routing, self.clock)

    # ---------- chain selection ----------
    def chain(self, mode: Mode) -> list[Provider]:
        out: list[Provider] = []
        for p in self.settings.list():
            if not p.enabled or not p.model:
                continue
            if mode is Mode.FREE_API_ONLY and p.kind is ProviderKind.OLLAMA:
                continue
            if mode is Mode.OLLAMA_ONLY and p.kind is not ProviderKind.OLLAMA:
                continue
            # Auto uses only tested providers; an explicit mode accepts untested ones.
            usable = (
                p.test_status in USABLE_IN_AUTO
                if mode is Mode.AUTO
                else p.test_status not in _HARD_FAILURES
            )
            if usable:
                out.append(p)
        return out

    # ---------- public calls ----------
    def chat(
        self,
        request: ChatRequest,
        mode: Mode | None = None,
        use_cache: bool | None = None,
        on_delta: OnDelta | None = None,
    ) -> ChatResponse:
        """With `on_delta` the answer is streamed: each piece is passed on as it arrives, and
        the same checks, retries and fallback apply until the first piece has been passed on."""
        use_cache = self.cfg.cache if use_cache is None else use_cache
        if use_cache and (hit := self.cache.get(request)) is not None:
            return hit
        mode = mode or self.settings.mode()
        chain = self.chain(mode)
        if not chain:
            raise NoProviderAvailable(
                f"No usable provider for mode '{mode.value}'. Open the Models page, enable a "
                "provider and run its connection test."
            )
        strategy = self.settings.get_setting(ROUTING_STRATEGY, self.cfg.routing.strategy)
        attempts: list[str] = []
        quota_only = True
        for provider in self.router.order(chain, strategy):
            try:
                resp = self.call_provider(
                    provider, request, self.cfg.request_timeout_s, on_delta=on_delta
                )
                if not resp.text.strip():
                    raise EmptyResponse(f"{provider.name}: returned an empty answer")
            except QuotaExhausted as exc:
                attempts.append(_attempt(provider, exc))
                continue
            except RateLimited as exc:
                # Too long to wait now: skip this provider until the time it asked for.
                self.router.cooldown(provider.name, exc.retry_after_s, "rate-limited")
                attempts.append(_attempt(provider, exc))
                continue
            except StreamInterrupted as exc:
                self.router.failure(provider.name, exc.code)
                raise
            except ProviderError as exc:
                quota_only = False
                self.router.failure(provider.name, exc.code)
                attempts.append(_attempt(provider, exc))
                logger.warning("provider %s failed: %s", provider.name, exc.code)
                continue
            self.router.success(provider.name, resp.latency_ms)
            resp = resp.model_copy(update={"attempts": tuple(attempts)})
            if use_cache:
                self.cache.put(request, resp)
            return resp
        detail = "; ".join(attempts)
        if quota_only:
            raise QuotaExhausted(f"All providers are out of quota or rate-limited. {detail}")
        raise NoProviderAvailable(f"Every provider failed. {detail}")

    def call_provider(
        self,
        provider: Provider,
        request: ChatRequest,
        timeout: float,
        retry: bool = True,
        on_delta: OnDelta | None = None,
    ) -> ChatResponse:
        """One provider, no cache, quota enforced. Raises the specific error on failure."""
        key = self.preflight(provider)
        est = self._estimate(request)
        attempts = self.cfg.max_attempts if retry else 1
        for attempt in range(1, attempts + 1):
            self._reserve(provider, est)
            started = time.perf_counter()
            try:
                resp = (
                    self._post_chat(provider, key, request, timeout)
                    if on_delta is None
                    else self._stream_chat(provider, key, request, timeout, on_delta)
                )
            except RateLimited as exc:
                self.limiter.settle(provider.name, est, 0)
                if exc.daily:
                    self.limiter.mark_exhausted(provider.name)
                    raise
                wait = (
                    exc.retry_after_s if exc.retry_after_s is not None else self._backoff(attempt)
                )
                if attempt == attempts or wait > self.cfg.max_wait_s:
                    raise
                self.clock.sleep(wait)
                continue
            except TransientProviderError:
                self.limiter.settle(provider.name, est, 0)
                if attempt == attempts:
                    raise
                self.clock.sleep(self._backoff(attempt))
                continue
            except ProviderError:
                self.limiter.settle(provider.name, est, 0)
                raise
            text, t_in, t_out = resp
            t_in = t_in if t_in is not None else est - request.max_tokens
            t_out = t_out if t_out is not None else self.tok.count(text)
            self.limiter.settle(provider.name, est, t_in + t_out)
            latency = int((time.perf_counter() - started) * 1000)
            logger.debug("usage %s in=%d out=%d %dms", provider.name, t_in, t_out, latency)
            return ChatResponse(
                text=text,
                provider=provider.name,
                model=provider.model,
                tokens_in=t_in,
                tokens_out=t_out,
                latency_ms=latency,
            )
        raise ProviderError(f"{provider.name}: no attempts made")  # unreachable by construction

    def list_models(self, provider: Provider, timeout: float) -> list[str] | None:
        """Model ids offered by a hosted provider; None when it has no list endpoint."""
        key = self.preflight(provider)
        try:
            r = self.client.get(
                P.models_url(provider), headers=P.auth_headers(key), timeout=timeout
            )
        except httpx.HTTPError as exc:
            raise ProviderUnreachable(
                f"{provider.name}: no answer from {provider.base_url}"
            ) from exc
        if r.status_code in (404, 405):
            return None
        if r.status_code >= 400:
            raise P.classify_error(r.status_code, dict(r.headers), r.text, provider.name)
        try:
            return P.parse_model_list(r.json())
        except ValueError:
            return None

    def embed(self, provider: Provider, model: str, texts: Sequence[str]) -> list[list[float]]:
        """Ollama embeddings (FR-20). Local, so no quota applies."""
        if provider.kind is not ProviderKind.OLLAMA:
            raise ProviderError("embeddings through the gateway are only offered for Ollama")
        self.preflight(provider)
        try:
            r = self.client.post(
                P.ollama_url(provider.base_url, "embed"),
                json={"model": model, "input": list(texts)},
                timeout=self.cfg.request_timeout_s,
            )
        except httpx.HTTPError as exc:
            raise ProviderUnreachable(f"Ollama is not running at {provider.base_url}") from exc
        if r.status_code >= 400:
            raise P.classify_error(r.status_code, dict(r.headers), r.text, provider.name)
        return [list(map(float, v)) for v in r.json().get("embeddings", [])]

    # ---------- internals ----------
    def preflight(self, provider: Provider) -> str | None:
        """Checks that run before any byte is sent: free flag, local address, key."""
        if not provider.free:
            raise ProviderNotFree(
                f"{provider.name} is not marked free; the gateway refuses non-free providers"
            )
        if provider.kind is ProviderKind.OLLAMA:
            if not P.is_local_url(provider.base_url) and not provider.allow_remote:
                raise RemoteOllamaNotConfirmed(
                    f"{provider.base_url} is not on this computer and was not confirmed"
                )
            return None
        key = self.secrets.resolve(provider.name, provider.api_key_env)
        if not key:
            raise KeyMissing(f"{provider.name}: no API key saved")
        return key

    def _estimate(self, request: ChatRequest) -> int:
        prompt = sum(self.tok.count(m.content) for m in request.messages)
        return prompt + request.max_tokens

    def _reserve(self, provider: Provider, est: int) -> None:
        wait = self.limiter.try_acquire(provider.name, provider.limits, est)
        if wait is not None and 0 < wait <= self.cfg.max_wait_s:
            self.clock.sleep(wait)
            wait = self.limiter.try_acquire(provider.name, provider.limits, est)
        if wait is None:
            raise QuotaExhausted(f"{provider.name}: daily quota used up")
        if wait > 0:
            raise RateLimited(f"{provider.name}: per-minute limit; free in {wait:.0f} s", wait)

    def _backoff(self, attempt: int) -> float:
        return self.cfg.backoff_base_s * (2 ** (attempt - 1)) * (1 + self._jitter())

    def _post_chat(
        self,
        provider: Provider,
        key: str | None,
        request: ChatRequest,
        timeout: float,
    ) -> tuple[str, int | None, int | None]:
        try:
            r = self.client.post(
                P.chat_url(provider),
                headers=P.auth_headers(key),
                json=P.chat_payload(provider.model, request),
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise _unreachable(provider, exc, timeout) from exc
        if r.status_code >= 400:
            raise P.classify_error(r.status_code, dict(r.headers), r.text, provider.name)
        try:
            return P.parse_chat(r.json())
        except ValueError as exc:
            raise ProviderError(f"{provider.name}: response was not JSON") from exc

    def _stream_chat(
        self,
        provider: Provider,
        key: str | None,
        request: ChatRequest,
        timeout: float,
        on_delta: OnDelta,
    ) -> tuple[str, int | None, int | None]:
        """Same request with `stream: true`. An error status arrives before any text, so it is
        classified exactly like the non-streamed call and can be retried or fallen back from."""
        parts: list[str] = []
        usage: dict[str, int] = {}
        try:
            with self.client.stream(
                "POST",
                P.chat_url(provider),
                headers=P.auth_headers(key),
                json=P.chat_payload(provider.model, request, stream=True),
                timeout=timeout,
            ) as r:
                if r.status_code >= 400:
                    body = r.read().decode("utf-8", errors="replace")
                    raise P.classify_error(r.status_code, dict(r.headers), body, provider.name)
                for line in r.iter_lines():
                    delta, seen = P.parse_stream_line(line)
                    usage = seen or usage
                    if delta:
                        parts.append(delta)
                        on_delta(delta)
        except httpx.HTTPError as exc:
            if parts:
                raise StreamInterrupted(
                    f"{provider.name}: the connection dropped while the answer was being written"
                ) from exc
            raise _unreachable(provider, exc, timeout) from exc
        return "".join(parts).strip(), usage.get("prompt_tokens"), usage.get("completion_tokens")


def _attempt(provider: Provider, exc: RagError) -> str:
    """One line per provider tried; most messages already start with the provider name."""
    named = exc.message.startswith(f"{provider.name}:")
    return exc.message if named else f"{provider.name}: {exc.message}"


def _unreachable(provider: Provider, exc: httpx.HTTPError, timeout: float) -> ProviderUnreachable:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderUnreachable(f"{provider.name}: timed out after {timeout:.0f} s")
    return ProviderUnreachable(f"{provider.name}: no answer from {provider.base_url}")
