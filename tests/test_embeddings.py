"""Tests for the local hashing embedder."""

from __future__ import annotations

import numpy as np
import pytest

from ragforge.embeddings.hashing import HashingEmbedder


def test_shape_and_dim() -> None:
    emb = HashingEmbedder(dim=256)
    vecs = emb.embed(["hello world", "another document here"])
    assert vecs.shape == (2, 256)
    assert emb.dim == 256


def test_vectors_are_normalised() -> None:
    emb = HashingEmbedder(dim=128)
    vecs = emb.embed(["the quick brown fox jumps over the lazy dog"])
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0, atol=1e-5)


def test_deterministic() -> None:
    emb = HashingEmbedder(dim=128)
    a = emb.embed_one("repeatable text input")
    b = emb.embed_one("repeatable text input")
    assert np.array_equal(a, b)


def test_similar_text_scores_higher_than_unrelated() -> None:
    emb = HashingEmbedder(dim=512)
    q = emb.embed_one("python programming language tutorial")
    close = emb.embed_one("a tutorial about the python programming language")
    far = emb.embed_one("the migratory patterns of arctic seabirds")
    assert float(q @ close) > float(q @ far)


def test_empty_string_is_zero_vector() -> None:
    emb = HashingEmbedder(dim=64)
    assert np.linalg.norm(emb.embed_one("")) == 0.0


def test_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)
