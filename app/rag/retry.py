"""Shared retry policy for OpenAI calls.

Retries only on transient failures (5xx, 429, network) — never on auth (401),
quota-exhausted (403/billing), or bad-request (400) errors, so we fail fast on
real bugs instead of looping pointlessly.
"""
from __future__ import annotations

from openai import APIConnectionError, APIStatusError, RateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):  # 429
        return True
    if isinstance(exc, APIConnectionError):  # network blip
        return True
    if isinstance(exc, APIStatusError):  # generic HTTP error
        status = getattr(exc, "status_code", None)
        return status is not None and status >= 500
    return False


# 5 attempts × exponential jitter (1s → 16s) ≈ up to ~30s total before giving up.
openai_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=1, max=16),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)
