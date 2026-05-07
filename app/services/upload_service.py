"""Upload orchestration: validate → extract → chunk → embed → persist → index BM25.

Pure business logic. Routers should only handle HTTP I/O and delegate here.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from app.core.config import settings
from app.core.exceptions import (
    AppError,
    EmbeddingServiceError,
    EmptyFileError,
    FileParseError,
    FileTooLargeError,
    NoExtractableTextError,
    TooManyFilesError,
    UnsupportedFileError,
)
from app.ingestion.chunker import chunk_blocks
from app.ingestion.constants import CHUNK_OVERLAP, CHUNK_SIZE
from app.ingestion.extractor import detect_kind, extract
from app.models.session import FileRecord, Session
from app.rag.embeddings import embed_texts
from app.rag.retriever import rebuild_bm25
from app.rag.store import collection_for

logger = logging.getLogger(__name__)


def _strip_none(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if v is not None}


def assert_capacity(session: Session, incoming: int) -> None:
    remaining = settings.max_files_per_session - len(session.files)
    if remaining <= 0:
        raise TooManyFilesError(
            f"Session already holds the maximum of {settings.max_files_per_session} files."
        )
    if incoming > remaining:
        raise TooManyFilesError(
            f"This upload would exceed the per-session limit "
            f"({settings.max_files_per_session}); {remaining} slot(s) left."
        )


def ingest_file(
    session: Session,
    *,
    filename: str,
    content_type: str | None,
    content: bytes,
) -> FileRecord:
    """Process one file end-to-end and append the resulting record to the session."""
    if not filename:
        filename = f"file_{uuid.uuid4().hex}"

    kind = detect_kind(filename, content_type)
    if kind is None:
        raise UnsupportedFileError(
            f"Unsupported file type for '{filename}'. Allowed: PDF, DOCX, image (PNG/JPEG/...)."
        )

    if len(content) == 0:
        raise EmptyFileError(f"File '{filename}' is empty.")
    if len(content) > settings.max_file_size_bytes:
        raise FileTooLargeError(
            f"File '{filename}' exceeds the {settings.max_file_size_bytes // (1024 * 1024)}MB limit."
        )

    try:
        blocks, page_count = extract(content, kind, max_pdf_pages=settings.max_pdf_pages)
    except AppError:
        raise
    except ValueError as exc:
        raise NoExtractableTextError(str(exc)) from exc
    except Exception as exc:
        logger.exception("Extraction failed for %s", filename)
        raise FileParseError(f"Failed to parse '{filename}'.") from exc

    if not blocks:
        raise NoExtractableTextError(
            f"No extractable text in '{filename}'. "
            "If it's a scan, ensure it's legible (we OCR English + Vietnamese)."
        )

    chunks = chunk_blocks(blocks, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise NoExtractableTextError(f"No chunks produced from '{filename}'.")

    file_id = uuid.uuid4().hex
    uploaded_at = datetime.now(UTC).isoformat()

    ids = [f"{file_id}_{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        _strip_none(
            {
                "file_id": file_id,
                "filename": filename,
                "mime_type": content_type or "",
                "kind": kind,
                "page": c.page,
                "section": c.section,
                "paragraph": c.paragraph,
                "chunk_index": c.chunk_index,
                "uploaded_at": uploaded_at,
            }
        )
        for c in chunks
    ]

    try:
        embeddings = embed_texts(documents)
    except Exception as exc:
        logger.exception("Embedding failed for %s", filename)
        raise EmbeddingServiceError("Embedding service unavailable.") from exc

    coll = collection_for(session.session_id)
    coll.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    record = FileRecord(
        file_id=file_id,
        filename=filename,
        mime_type=content_type or "",
        size_bytes=len(content),
        uploaded_at=uploaded_at,
        page_count=page_count if kind == "pdf" else None,
        chunk_count=len(chunks),
    )
    session.files.append(record)
    return record


def rebuild_session_index(session: Session) -> None:
    """Rebuild BM25 over every chunk in this session's Chroma collection."""
    coll = collection_for(session.session_id)
    if coll.count() == 0:
        rebuild_bm25(session, ids=[], texts=[])
        return
    data = coll.get(include=["documents"])
    rebuild_bm25(session, ids=list(data.get("ids", [])), texts=list(data.get("documents", [])))
