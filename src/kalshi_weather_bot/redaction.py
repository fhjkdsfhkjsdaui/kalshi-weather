"""Helpers for redacting sensitive values from logs and journal payloads."""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"

_SENSITIVE_KEY_RE = re.compile(
    r"(authorization|token|secret|signature|private[_-]?key|bearer|api[_-]?key)",
    re.IGNORECASE,
)
_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)
_AUTH_TOKEN_INLINE_RE = re.compile(
    r"(?i)\b(bearer)\s+[A-Za-z0-9\-._~+/]+=*",
)
_KEY_VALUE_SECRET_RE = re.compile(
    r"""(?ix)
    \b
    (
      authorization|
      token|
      secret|
      signature|
      private[_-]?key|
      bearer|
      api[_-]?key|
      kalshi-access-signature|
      kalshi-access-key
    )
    \s*[:=]\s*
    ([^\s,;]+)
    """
)


def sanitize_text(text: str) -> str:
    """Redact sensitive content embedded in plain text."""
    sanitized = _PRIVATE_KEY_BLOCK_RE.sub(REDACTED, text)
    sanitized = _AUTH_TOKEN_INLINE_RE.sub(r"\1 " + REDACTED, sanitized)
    sanitized = _KEY_VALUE_SECRET_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", sanitized)
    return sanitized


def sanitize_for_logging(value: Any) -> Any:
    """Recursively redact sensitive values in nested structures."""
    if isinstance(value, dict):
        sanitized: dict[Any, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if _SENSITIVE_KEY_RE.search(key_text):
                sanitized[key] = REDACTED
            else:
                sanitized[key] = sanitize_for_logging(child)
        return sanitized
    if isinstance(value, list):
        return [sanitize_for_logging(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_for_logging(item) for item in value)
    if isinstance(value, set):
        return {sanitize_for_logging(item) for item in value}
    if isinstance(value, str):
        return sanitize_text(value)
    return value

