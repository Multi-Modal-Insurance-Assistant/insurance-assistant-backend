"""Chat orchestration: precondition checks → hybrid retrieval → LLM → history append."""
from __future__ import annotations

import logging
from dataclasses import asdict

from app.core.exceptions import (
    EmptySessionError,
    LLMServiceError,
    NoDocumentsError,
    RetrievalServiceError,
)
from app.models.session import ChatTurn, Session
from app.rag.constants import HISTORY_LIMIT
from app.rag.llm import Answer, answer_with_context
from app.rag.retriever import hybrid_search
from app.rag.store import collection_for

logger = logging.getLogger(__name__)


def answer_question(session: Session, question: str) -> Answer:
    if not session.files:
        raise NoDocumentsError(
            "No documents uploaded yet. Please upload at least one file before asking questions."
        )
    if collection_for(session.session_id).count() == 0:
        raise EmptySessionError("Session has no indexed content.")

    try:
        hits = hybrid_search(session, question)
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise RetrievalServiceError("Retrieval service unavailable.") from exc

    try:
        result = answer_with_context(question, hits, list(session.history))
    except Exception as exc:
        logger.exception("LLM call failed")
        raise LLMServiceError("LLM service unavailable.") from exc

    session.history.append(ChatTurn(role="user", content=question))
    session.history.append(
        ChatTurn(
            role="assistant",
            content=result.answer,
            citations=[asdict(c) for c in result.citations],
        )
    )
    if len(session.history) > HISTORY_LIMIT:
        session.history = session.history[-HISTORY_LIMIT:]

    return result
