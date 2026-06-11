"""Tests for the lexical/BM25 hybrid reranker and two-stage retrieval."""

from __future__ import annotations

import pytest

from ragforge.config import Settings
from ragforge.pipeline import Pipeline
from ragforge.rerank import build_reranker
from ragforge.rerank.lexical import LexicalReranker
from ragforge.types import Chunk, Document, ScoredChunk


def _cand(doc_id: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=doc_id, doc_id=doc_id, text=text), score=score)


def test_rerank_empty_returns_empty() -> None:
    assert LexicalReranker().rerank("q", [], top_k=5) == []


def test_rerank_truncates_to_top_k() -> None:
    cands = [_cand(str(i), f"text {i}", 0.5) for i in range(10)]
    out = LexicalReranker().rerank("text", cands, top_k=3)
    assert len(out) == 3


def test_rerank_promotes_exact_term_match() -> None:
    # Embedding scores favour the wrong doc; BM25 should rescue the exact match.
    cands = [
        _cand("a", "general discussion of unrelated topics and filler", 0.90),
        _cand("b", "the cargo build tool compiles rust crates", 0.30),
    ]
    # A lexical-leaning blend lets the exact term match overtake the embedding lead.
    out = LexicalReranker(alpha=0.35).rerank("cargo build tool", cands, top_k=2)
    assert out[0].chunk.doc_id == "b"


def test_alpha_one_preserves_embedding_order() -> None:
    cands = [
        _cand("a", "no query terms here at all", 0.9),
        _cand("b", "cargo build tool", 0.1),
    ]
    out = LexicalReranker(alpha=1.0).rerank("cargo build tool", cands, top_k=2)
    assert out[0].chunk.doc_id == "a"  # pure embedding order


def test_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError):
        LexicalReranker(alpha=1.5)


def test_build_reranker_respects_flag() -> None:
    assert build_reranker(Settings(rerank_enabled=False)) is None
    assert isinstance(build_reranker(Settings(rerank_enabled=True)), LexicalReranker)


def test_pipeline_two_stage_retrieval_runs() -> None:
    settings = Settings(rerank_enabled=True, rerank_fetch_multiplier=4)
    pipeline = Pipeline(settings)
    pipeline.ingest(
        [
            Document(id="py", text="Python is a readable high-level programming language."),
            Document(id="rust", text="Rust uses the cargo build tool for memory safety."),
            Document(id="cook", text="Risotto needs arborio rice and constant stirring."),
        ]
    )
    assert pipeline.retriever.reranker is not None
    results = pipeline.retrieve("cargo build tool", top_k=2)
    assert results
    assert results[0].chunk.doc_id == "rust"
    assert len(results) <= 2


def test_pipeline_without_rerank_is_single_stage() -> None:
    pipeline = Pipeline(Settings(rerank_enabled=False))
    assert pipeline.retriever.reranker is None
