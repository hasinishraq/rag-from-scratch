"""Response cache keyed by the SHA-256 of provider-independent request content."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from rag.db import Database
from rag.types import ChatRequest, ChatResponse


def cache_key(request: ChatRequest) -> str:
    blob = json.dumps(
        {
            "purpose": request.purpose,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, request: ChatRequest) -> ChatResponse | None:
        row = self.db.one(
            "SELECT response_json FROM llm_cache WHERE key = ?", (cache_key(request),)
        )
        if row is None:
            return None
        resp = ChatResponse.model_validate_json(row["response_json"])
        return resp.model_copy(update={"cached": True, "latency_ms": 0})

    def put(self, request: ChatRequest, response: ChatResponse) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?)",
            (
                cache_key(request),
                response.model_dump_json(),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )

    def clear(self) -> int:
        row = self.db.one("SELECT COUNT(*) AS n FROM llm_cache")
        self.db.execute("DELETE FROM llm_cache")
        return int(row["n"]) if row else 0
