"""Reranking: a second stage that reorders first-pass retrieval candidates.

First-pass embedding retrieval is recall-oriented and cheap; a reranker improves
precision by rescoring a small candidate set with a stronger (or complementary)
signal. The default :class:`LexicalReranker` blends the embedding score with a
BM25 lexical score over the candidates — a hybrid signal that is fully local and
deterministic, so it needs no model or network.
"""

from __future__ import annotations

from ragforge.config import Settings
from ragforge.rerank.base import Reranker
from ragforge.rerank.lexical import LexicalReranker
from ragforge.rerank.llm_reranker import LLMReranker

__all__ = ["Reranker", "LexicalReranker", "LLMReranker", "build_reranker"]


def build_reranker(settings: Settings) -> Reranker | None:
    """Return a reranker if enabled in settings, else ``None`` (single-stage)."""
    if not settings.rerank_enabled:
        return None
    provider = settings.rerank_provider.lower()
    if provider == "lexical":
        return LexicalReranker(alpha=settings.rerank_alpha)
    if provider == "llm":
        from ragforge.llm import build_llm

        return LLMReranker(build_llm(settings))
    raise ValueError(f"Unknown rerank provider: {settings.rerank_provider!r}")
