from __future__ import annotations

from app.ingestion.chunker import TextBlock, chunk_blocks


def test_chunker_packs_short_blocks() -> None:
    blocks = [
        TextBlock(text="Hello world. " * 5, page=1, section=None, paragraph=None),
        TextBlock(text="Second paragraph. " * 5, page=1, section=None, paragraph=None),
    ]
    chunks = chunk_blocks(blocks, chunk_size=400, chunk_overlap=50)
    assert chunks
    assert all(c.page == 1 for c in chunks)
    assert all(0 < len(c.text) <= 400 for c in chunks)


def test_chunker_splits_oversized_block() -> None:
    big = TextBlock(text="A" * 2000, page=2, section=None, paragraph=None)
    chunks = chunk_blocks([big], chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 4
    assert all(c.page == 2 for c in chunks)
    assert all(len(c.text) <= 500 for c in chunks)


def test_chunker_preserves_section_paragraph_for_docx_blocks() -> None:
    blocks = [
        TextBlock(text="Intro paragraph", page=None, section="Section 1", paragraph=1),
        TextBlock(text="Body paragraph", page=None, section="Section 2 (Coverage)", paragraph=2),
    ]
    chunks = chunk_blocks(blocks, chunk_size=400, chunk_overlap=50)
    assert chunks[0].section == "Section 1"
    assert chunks[0].paragraph == 1
