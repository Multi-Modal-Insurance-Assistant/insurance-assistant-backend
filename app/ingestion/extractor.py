from __future__ import annotations

from app.ingestion.chunker import TextBlock
from app.ingestion.docx import extract_docx
from app.ingestion.image import extract_image
from app.ingestion.pdf import extract_pdf

_PDF_MIME = {"application/pdf"}
_DOCX_MIME = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}
_IMAGE_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/tiff", "image/bmp"}

_PDF_EXT = (".pdf",)
_DOCX_EXT = (".docx",)
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp")


def detect_kind(filename: str, content_type: str | None) -> str | None:
    ct = (content_type or "").lower().split(";")[0].strip()
    if ct in _PDF_MIME:
        return "pdf"
    if ct in _DOCX_MIME:
        return "docx"
    if ct in _IMAGE_MIME:
        return "image"
    name = (filename or "").lower()
    if name.endswith(_PDF_EXT):
        return "pdf"
    if name.endswith(_DOCX_EXT):
        return "docx"
    if name.endswith(_IMAGE_EXT):
        return "image"
    return None


def extract(content: bytes, kind: str, max_pdf_pages: int) -> tuple[list[TextBlock], int]:
    if kind == "pdf":
        return extract_pdf(content, max_pages=max_pdf_pages)
    if kind == "docx":
        return extract_docx(content)
    if kind == "image":
        return extract_image(content)
    raise ValueError(f"Unsupported kind: {kind}")
