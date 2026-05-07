from __future__ import annotations

import contextlib
import threading
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings

_client: Any | None = None
_lock = threading.Lock()


def get_chroma() -> Any:
    """Return a process-wide PersistentClient. Cosine space is set per-collection."""
    global _client
    with _lock:
        if _client is None:
            settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
            _client = chromadb.PersistentClient(
                path=str(settings.chroma_persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
        return _client


def _collection_name(session_id: str) -> str:
    safe = "".join(ch for ch in session_id if ch.isalnum())[:40] or "default"
    return f"sess_{safe}"


def collection_for(session_id: str) -> Any:
    return get_chroma().get_or_create_collection(
        name=_collection_name(session_id),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(session_id: str) -> None:
    name = _collection_name(session_id)
    client = get_chroma()
    with contextlib.suppress(Exception):
        # Collection may not exist; ignore.
        client.delete_collection(name=name)
