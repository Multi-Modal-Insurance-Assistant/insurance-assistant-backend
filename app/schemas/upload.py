from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class FileUploadInfo(BaseModel):
    file_id: str
    filename: str
    mime_type: str
    size_bytes: int
    page_count: int | None = None
    chunk_count: int
    uploaded_at: datetime


class UploadResponse(BaseModel):
    session_id: str
    files: list[FileUploadInfo]
    total_files_in_session: int
