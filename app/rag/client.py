from __future__ import annotations

import threading

from openai import OpenAI

from app.core.config import settings

_client: OpenAI | None = None
_lock = threading.Lock()


def get_openai_client() -> OpenAI:
    """Process-wide OpenAI client. We disable the SDK's built-in retry so our
    tenacity policy is the single source of truth for retries.
    """
    global _client
    with _lock:
        if _client is None:
            _client = OpenAI(api_key=settings.openai_api_key, max_retries=0)
        return _client
