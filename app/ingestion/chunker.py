from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TextBlock:
    """A logical text block tied to a citation anchor (page or section/paragraph)."""

    text: str
    page: int | None
    section: str | None
    paragraph: int | None


@dataclass
class Chunk:
    text: str
    page: int | None
    section: str | None
    paragraph: int | None
    chunk_index: int


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    step = max(1, max_chars - overlap)
    pos = 0
    while pos < len(text):
        out.append(text[pos : pos + max_chars])
        if pos + max_chars >= len(text):
            break
        pos += step
    return out


def _normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_blocks(blocks: list[TextBlock], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Pack consecutive blocks up to ~chunk_size chars; split blocks individually if they exceed it."""
    chunks: list[Chunk] = []
    idx = 0
    buffer: list[TextBlock] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal idx, buffer, buffer_len
        if not buffer:
            return
        text = _normalize("\n\n".join(b.text for b in buffer))
        if not text:
            buffer = []
            buffer_len = 0
            return
        head = buffer[0]
        chunks.append(
            Chunk(
                text=text,
                page=head.page,
                section=head.section,
                paragraph=head.paragraph,
                chunk_index=idx,
            )
        )
        idx += 1
        buffer = []
        buffer_len = 0

    for block in blocks:
        normalized = _normalize(block.text)
        if not normalized:
            continue
        if len(normalized) > chunk_size:
            flush()
            for piece in _split_long(normalized, chunk_size, chunk_overlap):
                chunks.append(
                    Chunk(
                        text=piece,
                        page=block.page,
                        section=block.section,
                        paragraph=block.paragraph,
                        chunk_index=idx,
                    )
                )
                idx += 1
            continue
        if buffer_len + len(normalized) > chunk_size and buffer:
            flush()
        buffer.append(
            TextBlock(text=normalized, page=block.page, section=block.section, paragraph=block.paragraph)
        )
        buffer_len += len(normalized) + 2

    flush()
    return chunks
