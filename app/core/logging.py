"""Centralized logging configuration with best-effort PII redaction.

We don't intentionally log prompt or response bodies — only filenames and
exception stack traces. The redactor is defense in depth: if PII (email,
phone, long digit ID) does sneak into a filename or exception message, it
gets masked before reaching the handler. It is best-effort, not a guarantee.
"""
from __future__ import annotations

import logging
import re

from app.core.config import settings

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Phone-like: optional leading +, 10-15 digits with optional space/dot/dash/paren separators.
_PHONE_RE = re.compile(r"\+?\d(?:[\s.\-()]*\d){9,14}")
# Bare digit runs of 9+ (CMND/CCCD/credit card/long numeric IDs). Run after phone.
_LONG_DIGITS_RE = re.compile(r"\b\d{9,}\b")


def _redact(text: str) -> str:
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _LONG_DIGITS_RE.sub("[ID]", text)
    return text


class RedactingFormatter(logging.Formatter):
    """Format the record normally, then mask PII in the final string.

    Running after super().format() means we see msg, args, exc_text and
    stack_info in one pass — pre-format filters miss exception tracebacks.
    """

    def format(self, record: logging.LogRecord) -> str:
        return _redact(super().format(record))


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    # Idempotent: re-running (e.g. in tests) replaces handlers cleanly.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    handler = logging.StreamHandler()
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
    # Quiet noisy third-party loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
