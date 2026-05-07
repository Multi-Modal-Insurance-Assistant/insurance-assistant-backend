from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Response

from app.api.deps import get_session
from app.core.config import settings
from app.core.session_store import get_session_store
from app.models.session import Session
from app.rag.store import reset_collection
from app.schemas.session import SessionResponse
from app.schemas.upload import FileUploadInfo

router = APIRouter(prefix="/session", tags=["session"])


@router.get("", response_model=SessionResponse)
def get_current_session(session: Session = Depends(get_session)) -> SessionResponse:
    return SessionResponse(
        session_id=session.session_id,
        files=[
            FileUploadInfo(
                file_id=f.file_id,
                filename=f.filename,
                mime_type=f.mime_type,
                size_bytes=f.size_bytes,
                page_count=f.page_count,
                chunk_count=f.chunk_count,
                uploaded_at=datetime.fromisoformat(f.uploaded_at),
            )
            for f in session.files
        ],
        file_count=len(session.files),
        max_files=settings.max_files_per_session,
    )


@router.delete("", status_code=204)
def delete_current_session(response: Response, session: Session = Depends(get_session)) -> None:
    """Wipe the session's vector collection and forget it. Frontend gets a fresh cookie next request."""
    reset_collection(session.session_id)
    get_session_store().delete(session.session_id)
    response.delete_cookie(settings.session_cookie_name, path="/")
