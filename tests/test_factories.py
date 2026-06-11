"""Tests for the configuration-driven factories and offline LLM selection."""

from __future__ import annotations

import pytest

from ragforge.config import Settings
from ragforge.embeddings import HashingEmbedder, build_embedder
from ragforge.llm import MockLLM, build_llm


def test_build_embedder_defaults_to_hashing() -> None:
    emb = build_embedder(Settings(embedding_provider="hashing", embedding_dim=128))
    assert isinstance(emb, HashingEmbedder)
    assert emb.dim == 128


def test_build_embedder_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        build_embedder(Settings(embedding_provider="does-not-exist"))


def test_build_llm_uses_mock_without_api_key() -> None:
    # No ANTHROPIC_API_KEY -> offline mock, never a live client.
    llm = build_llm(Settings(anthropic_api_key=None))
    assert isinstance(llm, MockLLM)


def test_settings_has_live_llm_flag() -> None:
    assert Settings(anthropic_api_key=None).has_live_llm is False
    assert Settings(anthropic_api_key="sk-test").has_live_llm is True
