"""Flags passages whose text reads like instructions to a model (prompt injection).

Detection only: such text is already sent inside <source> tags, declared to be data, with its
own tags neutralised. The flag tells the user, through a warning and the trace, that a document
tried to steer the model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_PATTERNS = [
    r"\bignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"\bdisregard (all |any |the )?(previous|prior|above|earlier|system)",
    r"\byou are now (in )?\w+",
    r"\b(developer|jailbreak|dan) mode\b",
    r"\b(reveal|print|show|leak) (the |your )?(system prompt|api key|password|secret)",
    r"</?\s*(system|source|instructions?)\s*>",
    r"^\s*(system|assistant)\s*:",
]
_INJECTION = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE | re.MULTILINE)


def looks_like_injection(text: str) -> bool:
    return _INJECTION.search(text) is not None


def suspicious_sources(texts: Sequence[str]) -> list[int]:
    """1-based numbers of the passages that contain instruction-like text."""
    return [i for i, t in enumerate(texts, start=1) if looks_like_injection(t)]
