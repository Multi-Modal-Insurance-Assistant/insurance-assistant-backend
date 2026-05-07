from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from app.api.deps import get_session
from app.models.session import Session
from app.schemas.chat import ChatRequest, ChatResponse, CitationOut
from app.services import chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, session: Session = Depends(get_session)) -> ChatResponse:
    result = await run_in_threadpool(chat_service.answer_question, session, req.question)
    return ChatResponse(
        session_id=session.session_id,
        answer=result.answer,
        citations=[
            CitationOut(
                index=c.index,
                file_id=c.file_id,
                filename=c.filename,
                location=c.location,
                chunk_id=c.chunk_id,
            )
            for c in result.citations
        ],
    )
