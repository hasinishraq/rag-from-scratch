"""Specific exceptions. The API maps each one to a JSON body {code, message} and an HTTP status."""

from __future__ import annotations


class RagError(Exception):
    code = "rag_error"
    http_status = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigError(RagError):
    code = "config_error"
    http_status = 500


class InvalidRequest(RagError):
    code = "invalid_request"
    http_status = 400


class NotFound(RagError):
    code = "not_found"
    http_status = 404


class Conflict(RagError):
    code = "conflict"
    http_status = 409


class Cancelled(RagError):
    """The client stopped waiting (closed the page or pressed Stop) while the answer was written."""

    code = "cancelled"
    http_status = 499


class Busy(RagError):
    code = "busy"
    http_status = 429


class PayloadTooLarge(RagError):
    code = "payload_too_large"
    http_status = 413


class IngestError(RagError):
    code = "ingest_error"
    http_status = 422


class ProviderError(RagError):
    """A provider call failed for a reason that is not one of the specific cases below."""

    code = "provider_error"
    http_status = 502


class TransientProviderError(ProviderError):
    """A 5xx answer: worth a retry with backoff before falling back."""

    code = "provider_error"
    http_status = 502


class EmptyResponse(ProviderError):
    """The provider answered 200 with no text (e.g. a reasoning model used its whole budget)."""

    code = "empty_response"
    http_status = 502


class StreamInterrupted(ProviderError):
    """The connection dropped after part of the answer was already shown: no retry, no fallback
    (another provider would start a second, different answer under the first)."""

    code = "stream_interrupted"
    http_status = 502


class ProviderNotFree(ProviderError):
    code = "provider_not_free"
    http_status = 403


class ProviderAuthError(ProviderError):
    code = "key_rejected"
    http_status = 502


class KeyMissing(ProviderError):
    code = "key_missing"
    http_status = 400


class ProviderUnreachable(ProviderError):
    code = "unreachable"
    http_status = 502


class ModelNotFound(ProviderError):
    code = "model_not_found"
    http_status = 502


class RemoteOllamaNotConfirmed(ProviderError):
    code = "remote_ollama_not_confirmed"
    http_status = 403


class RateLimited(ProviderError):
    code = "rate_limited"
    http_status = 503

    def __init__(self, message: str, retry_after_s: float | None = None, daily: bool = False):
        super().__init__(message)
        self.retry_after_s = retry_after_s
        self.daily = daily


class QuotaExhausted(RagError):
    code = "quota_exhausted"
    http_status = 503


class NoProviderAvailable(RagError):
    code = "no_provider_available"
    http_status = 503
