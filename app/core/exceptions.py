"""Application-level exceptions. Mapped to HTTP responses by a FastAPI exception handler."""
from __future__ import annotations


class AppError(Exception):
    """Base error for domain/business failures. Subclasses set status_code + code."""

    status_code: int = 400
    code: str = "app_error"

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


# --- Upload / ingestion ---
class UnsupportedFileError(AppError):
    status_code = 415
    code = "unsupported_file"


class EmptyFileError(AppError):
    status_code = 400
    code = "empty_file"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class TooManyFilesError(AppError):
    status_code = 429
    code = "too_many_files"


class PDFTooLongError(AppError):
    status_code = 422
    code = "pdf_too_long"


class NoExtractableTextError(AppError):
    status_code = 422
    code = "no_extractable_text"


class FileParseError(AppError):
    status_code = 500
    code = "file_parse_error"


# --- Chat ---
class NoDocumentsError(AppError):
    status_code = 400
    code = "no_documents"


class EmptySessionError(AppError):
    status_code = 400
    code = "empty_session"


# --- External services ---
class EmbeddingServiceError(AppError):
    status_code = 502
    code = "embedding_service_unavailable"


class LLMServiceError(AppError):
    status_code = 502
    code = "llm_service_unavailable"


class RetrievalServiceError(AppError):
    status_code = 502
    code = "retrieval_service_unavailable"
