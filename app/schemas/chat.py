from __future__ import annotations

from pydantic import BaseModel, Field


class CitationOut(BaseModel):
    index: int  # 1-based; matches the [#N] marker in the answer
    file_id: str
    filename: str
    location: str
    chunk_id: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[CitationOut]
