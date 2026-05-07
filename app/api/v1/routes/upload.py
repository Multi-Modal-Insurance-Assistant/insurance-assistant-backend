from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_session
from app.models.session import Session
from app.schemas.upload import FileUploadInfo, UploadResponse
from app.services import upload_service

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("", response_model=UploadResponse)
async def upload_files(
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_session),
) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    upload_service.assert_capacity(session, incoming=len(files))

    uploaded: list[FileUploadInfo] = []
    for upload in files:
        content = await upload.read()
        record = upload_service.ingest_file(
            session,
            filename=upload.filename or "",
            content_type=upload.content_type,
            content=content,
        )
        uploaded.append(
            FileUploadInfo(
                file_id=record.file_id,
                filename=record.filename,
                mime_type=record.mime_type,
                size_bytes=record.size_bytes,
                page_count=record.page_count,
                chunk_count=record.chunk_count,
                uploaded_at=datetime.fromisoformat(record.uploaded_at),
            )
        )

    upload_service.rebuild_session_index(session)

    return UploadResponse(
        session_id=session.session_id,
        files=uploaded,
        total_files_in_session=len(session.files),
    )
