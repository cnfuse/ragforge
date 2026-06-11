"""Tests for the in-memory vector store and the retriever."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ragforge.embeddings.hashing import HashingEmbedder
from ragforge.ingestion.chunking import chunk_document
from ragforge.retrieval.retriever import Retriever
from ragforge.store.memory import InMemoryVectorStore
from ragforge.types import Chunk, Document

CORPUS = {
    "py": "Python is a high-level programming language known for readable syntax.",
    "rust": "Rust is a systems programming language focused on memory safety.",
    "cook": "A classic risotto needs arborio rice, stock, and constant stirring.",
}


def _build_retriever(dim: int = 256) -> Retriever:
    emb = HashingEmbedder(dim=dim)
    store = InMemoryVectorStore(dim=dim)
    retriever = Retriever(emb, store)
    chunks: list[Chunk] = []
    for doc_id, text in CORPUS.items():
        chunks.extend(chunk_document(Document(id=doc_id, text=text), chunk_size=400))
    retriever.index(chunks)
    return retriever


def test_empty_store_returns_nothing() -> None:
    store = InMemoryVectorStore(dim=32)
    assert store.search(np.zeros(32, dtype=np.float32), top_k=5) == []
    assert len(store) == 0


def test_dimension_mismatch_raises() -> None:
    store = InMemoryVectorStore(dim=16)
    chunk = Chunk(id="c", doc_id="d", text="x")
    with pytest.raises(ValueError):
        store.add([chunk], np.zeros((1, 8), dtype=np.float32))


def test_retrieval_ranks_relevant_document_first() -> None:
    retriever = _build_retriever()
    results = retriever.retrieve("memory safe systems language", top_k=3)
    assert results
    assert results[0].chunk.doc_id == "rust"
    # Scores must come back sorted descending.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_capped_at_store_size() -> None:
    retriever = _build_retriever()
    results = retriever.retrieve("programming", top_k=99)
    assert len(results) == len(retriever.store)


def test_blank_query_returns_empty() -> None:
    retriever = _build_retriever()
    assert retriever.retrieve("   ", top_k=3) == []


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    retriever = _build_retriever()
    path = tmp_path / "index.json"
    retriever.store.save(path)  # type: ignore[attr-defined]

    loaded = InMemoryVectorStore.load(path)
    assert len(loaded) == len(retriever.store)
    emb = HashingEmbedder(dim=loaded.dim)
    results = loaded.search(emb.embed_one("readable programming language"), top_k=1)
    assert results[0].chunk.doc_id == "py"
