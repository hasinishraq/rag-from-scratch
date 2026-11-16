"""Server-sent events for POST /query/stream: the passages as soon as they are chosen, the answer
piece by piece, then the complete Answer (or one error). The pipeline runs in its own thread and
this generator relays what it reports, so a slow reader never blocks the model call."""

from __future__ import annotations

import contextvars
import json
import logging
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

from rag.errors import Cancelled, RagError
from rag.security import redact
from rag.types import Answer, StreamEvent

logger = logging.getLogger(__name__)

# A comment line this often keeps proxies and browsers from closing a quiet connection
# (a provider may think for a while before its first word).
KEEPALIVE_S = 15.0
SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}

RunAnswer = Callable[[Callable[[StreamEvent], None]], Answer]


def frame(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _payload(event: StreamEvent) -> dict[str, Any]:
    if event.kind == "sources":
        return {"sources": [s.model_dump(mode="json") for s in event.sources]}
    return {"text": event.text}


def answer_events(run: RunAnswer, keepalive_s: float = KEEPALIVE_S) -> Iterator[bytes]:
    """Yields SSE frames: `sources`, `delta`..., then `done` or `error`. Closing the generator
    (the client went away or pressed Stop) cancels the answer at its next piece of text."""
    frames: queue.Queue[bytes | None] = queue.Queue()
    stopped = threading.Event()

    def on_event(event: StreamEvent) -> None:
        if stopped.is_set():
            raise Cancelled("the reader stopped waiting before the answer was complete")
        frames.put(frame(event.kind, _payload(event)))

    def work() -> None:
        try:
            frames.put(frame("done", run(on_event).model_dump(mode="json")))
        except RagError as exc:
            frames.put(frame("error", {"code": exc.code, "message": redact(exc.message)}))
        except Exception as exc:  # the stream must always end with a final event
            logger.exception("unhandled error while streaming: %s", type(exc).__name__)
            frames.put(frame("error", {"code": "internal_error", "message": "internal error"}))
        finally:
            frames.put(None)

    # Copy the context so log lines from the worker carry the request id.
    context = contextvars.copy_context()
    threading.Thread(target=context.run, args=(work,), name="answer-stream", daemon=True).start()
    try:
        while True:
            try:
                item = frames.get(timeout=keepalive_s)
            except queue.Empty:
                yield b": keep-alive\n\n"
                continue
            if item is None:
                return
            yield item
    finally:
        stopped.set()
