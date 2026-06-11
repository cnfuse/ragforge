"""Document ingestion: loading raw sources and splitting them into chunks."""

from __future__ import annotations

from ragforge.ingestion.chunking import chunk_document, chunk_text
from ragforge.ingestion.loaders import load_path, load_text

__all__ = ["chunk_document", "chunk_text", "load_path", "load_text"]
