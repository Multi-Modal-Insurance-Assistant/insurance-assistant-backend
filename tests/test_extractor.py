from __future__ import annotations

from app.ingestion.extractor import detect_kind


def test_detect_kind_by_mime() -> None:
    assert detect_kind("a.pdf", "application/pdf") == "pdf"
    assert (
        detect_kind(
            "a.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        == "docx"
    )
    assert detect_kind("a.png", "image/png") == "image"


def test_detect_kind_falls_back_to_extension() -> None:
    assert detect_kind("policy.PDF", None) == "pdf"
    assert detect_kind("scan.JPEG", "application/octet-stream") == "image"


def test_detect_kind_unknown() -> None:
    assert detect_kind("notes.txt", "text/plain") is None
