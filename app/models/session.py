"""Domain models for an in-flight chat session. Pure dataclasses, no infra dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi


@dataclass
class FileRecord:
    file_id: str
    filename: str
    mime_type: str
    size_bytes: int
    uploaded_at: str  # ISO 8601
    page_count: int | None
    chunk_count: int


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str
    citations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Session:
    session_id: str
    created_at: float
    last_seen_at: float
    files: list[FileRecord] = field(default_factory=list)
    history: list[ChatTurn] = field(default_factory=list)
    bm25: BM25Okapi | None = None
    bm25_doc_ids: list[str] = field(default_factory=list)
