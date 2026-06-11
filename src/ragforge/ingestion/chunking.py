"""Paragraph-aware character chunking with overlap.

The splitter favours natural boundaries (paragraph breaks, then sentence
boundaries) and only falls back to a hard character cut when a single span
exceeds the target size. Overlap preserves context across chunk edges so a fact
that straddles a boundary is still retrievable.
"""

from __future__ import annotations

import re

from ragforge.types import Chunk, Document

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _split_units(text: str) -> list[str]:
    """Break text into paragraphs, then sentences within long paragraphs."""
    units: list[str] = []
    for para in _PARAGRAPH.split(text):
        para = para.strip()
        if not para:
            continue
        units.extend(s for s in _SENTENCE.split(para) if s)
    return units


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split ``text`` into overlapping chunks of roughly ``chunk_size`` chars.

    Args:
        text: Raw document text.
        chunk_size: Target maximum characters per chunk.
        overlap: Number of trailing characters to repeat at the start of the
            next chunk, preserving cross-boundary context.

    Returns:
        A list of chunk strings in document order.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    for unit in _split_units(text):
        # A single oversized unit is hard-split into chunk_size windows.
        if len(unit) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(unit), chunk_size - overlap):
                chunks.append(unit[start : start + chunk_size])
            continue

        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {unit}".strip() if tail else unit

    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def chunk_document(
    doc: Document, *, chunk_size: int = 800, overlap: int = 120
) -> list[Chunk]:
    """Chunk a :class:`Document`, propagating its id and metadata to each chunk."""
    pieces = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(
            id=f"{doc.id}::{i}",
            doc_id=doc.id,
            text=piece,
            ordinal=i,
            metadata=dict(doc.metadata),
        )
        for i, piece in enumerate(pieces)
    ]
