from __future__ import annotations

import contextlib
import io

from docx import Document

from app.ingestion.chunker import TextBlock


def extract_docx(content: bytes) -> tuple[list[TextBlock], int]:
    """Extract DOCX paragraphs and table rows. DOCX pagination is unstable, so we cite Section/Paragraph."""
    doc = Document(io.BytesIO(content))
    blocks: list[TextBlock] = []
    section_idx = 1
    paragraph_idx = 0
    current_heading: str | None = None

    for para in doc.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue
        paragraph_idx += 1

        style_name = ""
        with contextlib.suppress(Exception):
            style_name = (para.style.name or "").lower() if para.style else ""

        if style_name.startswith("heading"):
            section_idx += 1
            current_heading = text

        section_label = f"Section {section_idx}"
        if current_heading:
            section_label = f"Section {section_idx} ({current_heading})"

        blocks.append(
            TextBlock(text=text, page=None, section=section_label, paragraph=paragraph_idx)
        )

    for tbl_idx, table in enumerate(doc.tables, start=1):
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if not row_text:
                continue
            paragraph_idx += 1
            blocks.append(
                TextBlock(
                    text=row_text,
                    page=None,
                    section=f"Table {tbl_idx}",
                    paragraph=paragraph_idx,
                )
            )

    return blocks, paragraph_idx
