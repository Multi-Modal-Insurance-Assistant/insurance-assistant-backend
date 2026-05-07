from __future__ import annotations

import io

import pytesseract
from PIL import Image

from app.core.config import settings
from app.ingestion.chunker import TextBlock
from app.ingestion.constants import OCR_MIN_LONG_EDGE_PX, OCR_PSM


def _prepare_for_ocr(img: Image.Image) -> Image.Image:
    """Normalise mode + upscale tiny images so Tesseract sees larger glyphs.

    Empirically, posters / ID cards / table cells under ~2000px on the long
    edge OCR poorly with default settings. A simple Lanczos upscale recovers
    a lot — phone numbers, table cells, dense captions all improve.
    """
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    long_edge = max(img.size)
    if long_edge < OCR_MIN_LONG_EDGE_PX:
        scale = OCR_MIN_LONG_EDGE_PX / long_edge
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def ocr_image_bytes(content: bytes, lang: str | None = None) -> str:
    """Run Tesseract OCR on raw image bytes with upscale + tuned PSM."""
    img = Image.open(io.BytesIO(content))
    img = _prepare_for_ocr(img)
    return pytesseract.image_to_string(
        img,
        lang=lang or settings.ocr_languages,
        config=f"--psm {OCR_PSM}",
    )


def extract_image(content: bytes) -> tuple[list[TextBlock], int]:
    text = ocr_image_bytes(content).strip()
    if not text:
        return [], 1
    return [TextBlock(text=text, page=1, section=None, paragraph=None)], 1
