from __future__ import annotations

import logging

import fitz  # PyMuPDF

from app.core.exceptions import PDFTooLongError
from app.ingestion.chunker import TextBlock
from app.ingestion.constants import OCR_FALLBACK_THRESHOLD_CHARS, PDF_OCR_DPI
from app.ingestion.image import ocr_image_bytes

logger = logging.getLogger(__name__)


def extract_pdf(content: bytes, max_pages: int) -> tuple[list[TextBlock], int]:
    """Extract per-page text via PyMuPDF; OCR-fallback any page that looks scanned."""
    doc = fitz.open(stream=content, filetype="pdf")
    try:
        page_count = doc.page_count
        if page_count > max_pages:
            raise PDFTooLongError(
                f"PDF has {page_count} pages; the maximum is {max_pages}."
            )

        blocks: list[TextBlock] = []
        for page_index in range(page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text").strip()

            if len(text) < OCR_FALLBACK_THRESHOLD_CHARS:
                try:
                    pix = page.get_pixmap(dpi=PDF_OCR_DPI)
                    img_bytes = pix.tobytes("png")
                    ocr_text = ocr_image_bytes(img_bytes).strip()
                    if len(ocr_text) > len(text):
                        text = ocr_text
                except Exception:
                    logger.exception("OCR fallback failed for page %s", page_index + 1)

            if text:
                blocks.append(
                    TextBlock(text=text, page=page_index + 1, section=None, paragraph=None)
                )

        return blocks, page_count
    finally:
        doc.close()
