"""Tests for the BM25 sparse index and hybrid (RRF) retrieval."""

from __future__ import annotations

from pathlib import Path

from ragforge.config import Settings
from ragforge.pipeline import Pipeline
from ragforge.store.sparse import BM25Index
from ragforge.types import Chunk, Document

CORPUS = [
    Document(id="py", text="Python is a readable high-level programming language."),
    Document(id="rust", text="Rust uses the cargo build tool and guarantees memory safety."),
    Document(id="cook", text="Risotto needs arborio rice and constant stirring."),
]


def _chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(id=doc_id, doc_id=doc_id, text=text)


# --- BM25 index -----------------------------------------------------------


def test_bm25_empty_index() -> None:
    idx = BM25Index()
    assert len(idx) == 0
    assert idx.search("anything", top_k=3) == []


def test_bm25_ranks_term_match_first() -> None:
    idx = BM25Index()
    idx.add([_chunk(d.id, d.text) for d in CORPUS])
    hits = idx.search("cargo build tool", top_k=3)
    assert hits
    assert hits[0].chunk.doc_id == "rust"


def test_bm25_only_returns_matches() -> None:
    idx = BM25Index()
    idx.add([_chunk(d.id, d.text) for d in CORPUS])
    # A query term present in only one doc should not surface unrelated docs.
    hits = idx.search("risotto", top_k=5)
    assert [h.chunk.doc_id for h in hits] == ["cook"]


# --- hybrid retrieval -----------------------------------------------------


def test_pipeline_hybrid_enabled_builds_sparse() -> None:
    pipeline = Pipeline(Settings(hybrid_enabled=True))
    pipeline.ingest(CORPUS)
    assert pipeline.retriever.sparse is not None
    assert len(pipeline.retriever.sparse) == len(CORPUS)


def test_hybrid_retrieval_finds_term_specific_doc() -> None:
    pipeline = Pipeline(Settings(hybrid_enabled=True))
    pipeline.ingest(CORPUS)
    results = pipeline.retrieve("cargo build tool", top_k=2)
    assert results
    assert results[0].chunk.doc_id == "rust"


def test_hybrid_disabled_has_no_sparse() -> None:
    pipeline = Pipeline(Settings(hybrid_enabled=False))
    assert pipeline.retriever.sparse is None


def test_hybrid_rebuilt_from_loaded_index(tmp_path: Path) -> None:
    # Ingest + persist with a plain pipeline, then load with hybrid enabled:
    # the sparse index must be rebuilt from the loaded chunks.
    build = Pipeline(Settings())
    build.ingest(CORPUS)
    index = tmp_path / "idx.json"
    build.save_index(index)

    loaded = Pipeline.from_index(index, Settings(hybrid_enabled=True))
    assert loaded.retriever.sparse is not None
    assert len(loaded.retriever.sparse) == len(build.store)
    results = loaded.retrieve("cargo build tool", top_k=2)
    assert results[0].chunk.doc_id == "rust"
