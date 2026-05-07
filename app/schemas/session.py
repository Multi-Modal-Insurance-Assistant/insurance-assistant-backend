from __future__ import annotations

from pydantic import BaseModel

from app.schemas.upload import FileUploadInfo


class SessionResponse(BaseModel):
    session_id: str
    files: list[FileUploadInfo]
    file_count: int
    max_files: int
