"""Tests for Maximal Marginal Relevance diversification."""

from __future__ import annotations

import numpy as np
import pytest

from ragforge.config import Settings
from ragforge.pipeline import Pipeline
from ragforge.retrieval.mmr import mmr_select
from ragforge.types import Chunk, Document, ScoredChunk


def _cand(doc_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=doc_id, doc_id=doc_id, text=doc_id), score=score)


def test_mmr_empty() -> None:
    assert mmr_select([], np.empty((0, 3)), top_k=3) == []


def test_mmr_truncates_to_top_k() -> None:
    cands = [_cand(str(i), 1.0 - i * 0.1) for i in range(5)]
    vecs = np.eye(5, dtype=np.float32)  # all orthogonal
    out = mmr_select(cands, vecs, top_k=3, lambda_=0.5)
    assert len(out) == 3


def test_mmr_first_pick_is_most_relevant() -> None:
    cands = [_cand("a", 0.2), _cand("b", 0.9), _cand("c", 0.5)]
    vecs = np.eye(3, dtype=np.float32)
    out = mmr_select(cands, vecs, top_k=3, lambda_=0.5)
    assert out[0].chunk.doc_id == "b"


def test_mmr_avoids_redundant_second_pick() -> None:
    # a and b are near-duplicates (same vector); c is orthogonal and slightly
    # less relevant. Diversity-leaning MMR should pick c second, not b.
    cands = [_cand("a", 1.0), _cand("b", 0.95), _cand("c", 0.8)]
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = mmr_select(cands, vecs, top_k=2, lambda_=0.3)
    assert [s.chunk.doc_id for s in out] == ["a", "c"]


def test_mmr_lambda_one_is_pure_relevance() -> None:
    cands = [_cand("a", 1.0), _cand("b", 0.95), _cand("c", 0.8)]
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = mmr_select(cands, vecs, top_k=2, lambda_=1.0)
    assert [s.chunk.doc_id for s in out] == ["a", "b"]  # ignores redundancy


def test_mmr_invalid_lambda() -> None:
    with pytest.raises(ValueError):
        mmr_select([_cand("a", 1.0)], np.eye(1), top_k=1, lambda_=2.0)


def test_pipeline_mmr_enabled_runs() -> None:
    pipeline = Pipeline(Settings(mmr_enabled=True, mmr_lambda=0.5))
    pipeline.ingest(
        [
            Document(id="py", text="Python is a readable programming language."),
            Document(id="rust", text="Rust is a systems language with memory safety."),
            Document(id="go", text="Go is a compiled language with goroutines."),
        ]
    )
    assert pipeline.retriever.mmr_enabled
    results = pipeline.retrieve("programming language", top_k=2)
    assert len(results) <= 2
    # Distinct documents (diversification should not return duplicates).
    assert len({r.chunk.doc_id for r in results}) == len(results)
