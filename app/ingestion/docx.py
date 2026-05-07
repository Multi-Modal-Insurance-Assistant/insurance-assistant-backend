from __future__ import annotations

import contextlib
import io
import re
from typing import Iterator

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingestion.chunker import TextBlock

_ANCHOR_MAX_CHARS = 60


def _snippet_anchor(text: str) -> str:
    """First sentence (or first ~60 chars) of `text`, suitable for Ctrl-F in Word.

    Used as the citation anchor for content that has no preceding heading
    (intro/preamble paragraphs, headerless flyers, simple contracts).
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    # Prefer first sentence if short enough.
    sentence = re.split(r"(?<=[.!?])\s", cleaned, maxsplit=1)[0]
    candidate = sentence if len(sentence) <= _ANCHOR_MAX_CHARS else cleaned
    if len(candidate) <= _ANCHOR_MAX_CHARS:
        return candidate
    return candidate[: _ANCHOR_MAX_CHARS - 1].rstrip() + "…"


def _iter_body_blocks(doc) -> Iterator[Paragraph | Table]:
    """Yield Paragraph / Table objects in the order they appear in the document body.

    python-docx exposes paragraphs and tables on separate collections, which loses
    their relative order. Walking the body XML preserves it so a table inherits
    the heading context of whatever section it sits under.
    """
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def extract_docx(content: bytes) -> tuple[list[TextBlock], int]:
    """Extract DOCX paragraphs and table rows.

    Citation anchor philosophy: DOCX pagination is unstable, so instead of
    inventing Section/Paragraph numbers the user can't easily find, we cite the
    nearest preceding **heading text** — something the user can paste straight
    into Word's Find dialog. For content that sits before any heading (a one-page
    flyer, a doc preamble), we fall back to a short snippet of the paragraph
    itself, which is also Ctrl-F-able. Either way `section` is a self-describing
    string, never a bare ordinal.
    """
    doc = Document(io.BytesIO(content))
    blocks: list[TextBlock] = []
    paragraph_idx = 0
    current_heading: str | None = None

    for item in _iter_body_blocks(doc):
        if isinstance(item, Paragraph):
            text = (item.text or "").strip()
            if not text:
                continue
            paragraph_idx += 1

            style_name = ""
            with contextlib.suppress(Exception):
                style_name = (item.style.name or "").lower() if item.style else ""

            is_heading = style_name.startswith("heading")
            if is_heading:
                current_heading = text

            # Heading paragraphs anchor on their own text; sub-paragraphs anchor on
            # the nearest heading; preamble (no heading yet) anchors on its own snippet.
            anchor = current_heading if (current_heading and not is_heading) else (
                text if is_heading else _snippet_anchor(text)
            )
            blocks.append(
                TextBlock(
                    text=text,
                    page=None,
                    section=anchor,
                    paragraph=paragraph_idx,
                )
            )
        else:  # Table
            for row in item.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if not row_text:
                    continue
                paragraph_idx += 1
                anchor = current_heading or _snippet_anchor(row_text)
                blocks.append(
                    TextBlock(
                        text=row_text,
                        page=None,
                        section=anchor,
                        paragraph=paragraph_idx,
                    )
                )

    return blocks, paragraph_idx
