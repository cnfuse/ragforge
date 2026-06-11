"""Tests for paragraph-aware chunking."""

from __future__ import annotations

import pytest

from ragforge.ingestion.chunking import chunk_document, chunk_text
from ragforge.types import Document


def test_short_text_is_single_chunk() -> None:
    chunks = chunk_text("A short sentence.", chunk_size=800, overlap=100)
    assert chunks == ["A short sentence."]


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("   \n\n  ") == []


def test_chunks_respect_size_budget() -> None:
    text = ". ".join(f"sentence number {i} has some words" for i in range(200))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    # Allow a little slack for the joining space, never wildly over budget.
    assert all(len(c) <= 200 + 40 for c in chunks)


def test_oversized_unit_is_hard_split() -> None:
    text = "x" * 1000  # one giant token, no natural boundary
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) >= 4
    assert all(len(c) <= 300 for c in chunks)


def test_overlap_preserves_context() -> None:
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=120, overlap=40)
    assert len(chunks) >= 2
    # Consecutive chunks should share some trailing/leading content.
    assert any(
        chunks[i].split()[-1] in chunks[i + 1] for i in range(len(chunks) - 1)
    )


@pytest.mark.parametrize("bad", [(0, 0), (100, 100), (100, -1), (100, 150)])
def test_invalid_parameters_raise(bad: tuple[int, int]) -> None:
    size, overlap = bad
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=size, overlap=overlap)


def test_chunk_document_propagates_metadata_and_ids() -> None:
    doc = Document(id="doc1", text="alpha beta. " * 100, metadata={"k": "v"})
    chunks = chunk_document(doc, chunk_size=150, overlap=30)
    assert all(c.doc_id == "doc1" for c in chunks)
    assert all(c.metadata == {"k": "v"} for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert chunks[0].id == "doc1::0"
