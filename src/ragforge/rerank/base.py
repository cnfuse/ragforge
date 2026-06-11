"""Reranker protocol."""

from __future__ import annotations

from typing import Protocol

from ragforge.types import ScoredChunk


class Reranker(Protocol):
    """Reorders first-pass candidates and returns the best ``top_k``.

    Implementations receive the candidates with their first-pass scores and the
    original query, and return a re-sorted, possibly re-scored list truncated to
    ``top_k``.
    """

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...
