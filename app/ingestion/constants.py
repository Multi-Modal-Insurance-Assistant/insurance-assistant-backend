"""Ingestion-tier hyperparameters. Behavior-affecting; change via code review, not env."""
from __future__ import annotations

# Chunker
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 120

# PDF OCR fallback for scanned/image-only pages
OCR_FALLBACK_THRESHOLD_CHARS: int = 20
PDF_OCR_DPI: int = 200

# Pre-OCR image upscale: small images (long-edge < target) get scaled up
# before Tesseract runs. Bigger glyphs → much better recognition for
# posters, ID cards, table cells and small captions.
OCR_MIN_LONG_EDGE_PX: int = 2000
# Tesseract page-segmentation mode. 6 = "uniform block of text" — works
# better than the default (3=auto) for posters and dense single-column
# layouts; still acceptable for tables.
OCR_PSM: int = 6
