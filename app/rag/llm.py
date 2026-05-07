from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.models.session import ChatTurn
from app.rag.client import get_openai_client
from app.rag.constants import LLM_MAX_OUTPUT_TOKENS, LLM_TEMPERATURE, MAX_HISTORY_TURNS
from app.rag.prompts import load_answer_prompt
from app.rag.retriever import Hit
from app.rag.retry import openai_retry


@dataclass
class Citation:
    index: int  # 1-based; matches the [#N] marker the LLM emits
    file_id: str
    filename: str
    location: str
    chunk_id: str


@dataclass
class Answer:
    answer: str
    citations: list[Citation]


def _format_location(meta: dict[str, Any]) -> str:
    """Build a user-friendly citation anchor.

    PDFs cite by page (stable). DOCX cite by the nearest heading text (or, for
    preamble content, a short snippet of the paragraph itself) — that's the most
    actionable string for the user, since they can paste it straight into Word's
    Find dialog. We deliberately do NOT surface DOCX paragraph indices to the
    end user: they're meaningless without an anchor and look like noise.
    """
    page = meta.get("page")
    if page:
        return f"Page {page}"
    section = meta.get("section")
    if section:
        return str(section)
    return "Unknown location"


def _format_context(hits: list[Hit]) -> tuple[str, list[Citation]]:
    blocks: list[str] = []
    citations: list[Citation] = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.metadata or {}
        filename = str(meta.get("filename", "unknown"))
        location = _format_location(meta)
        blocks.append(f"[#{i}] {filename} — {location}\n{hit.text}")
        citations.append(
            Citation(
                index=i,
                file_id=str(meta.get("file_id", "")),
                filename=filename,
                location=location,
                chunk_id=hit.chunk_id,
            )
        )
    return "\n\n---\n\n".join(blocks), citations


def _format_history(history: list[ChatTurn], max_turns: int = MAX_HISTORY_TURNS) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    trimmed = history[-max_turns * 2 :]
    for turn in trimmed:
        role = "user" if turn.role == "user" else "assistant"
        out.append({"role": role, "content": turn.content})
    return out


def _filter_used_citations(answer_text: str, citations: list[Citation]) -> list[Citation]:
    used: list[Citation] = []
    seen: set[str] = set()
    for citation in citations:
        if f"[#{citation.index}]" in answer_text and citation.chunk_id not in seen:
            used.append(citation)
            seen.add(citation.chunk_id)
    return used


@openai_retry
def answer_with_context(question: str, hits: list[Hit], history: list[ChatTurn]) -> Answer:
    prompt = load_answer_prompt()

    if not hits:
        return Answer(answer=prompt.fallback_no_context, citations=[])

    context_str, citations = _format_context(hits)
    user_message = f"<context>\n{context_str}\n</context>\n\nQuestion: {question}"

    messages: list[dict[str, str]] = [{"role": "system", "content": prompt.system_instruction}]
    messages.extend(_format_history(history))
    messages.append({"role": "user", "content": user_message})

    client = get_openai_client()
    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_OUTPUT_TOKENS,
    )

    text = (response.choices[0].message.content or "").strip()
    if not text:
        text = prompt.fallback_no_context

    return Answer(answer=text, citations=_filter_used_citations(text, citations))
