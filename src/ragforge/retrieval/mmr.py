"""Maximal Marginal Relevance (MMR) diversification.

Top-k by pure relevance often returns near-duplicate passages, wasting the
context budget. MMR greedily selects passages that are relevant to the query yet
*dissimilar* to those already chosen, trading a little relevance for coverage.
Each pick maximises::

    lambda * relevance(c) - (1 - lambda) * max_sim(c, already_selected)

``lambda = 1`` reduces to plain relevance ranking; lower values push for
diversity. Operates on candidate embeddings (cosine == dot product, since
vectors are L2-normalised), so it is local and deterministic.
"""

from __future__ import annotations

import numpy as np

from ragforge.types import ScoredChunk


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def mmr_select(
    candidates: list[ScoredChunk],
    vectors: np.ndarray,
    *,
    top_k: int,
    lambda_: float = 0.5,
) -> list[ScoredChunk]:
    """Select up to ``top_k`` candidates by Maximal Marginal Relevance.

    Args:
        candidates: First-pass results, carrying their relevance score.
        vectors: ``(len(candidates), dim)`` L2-normalised embeddings, row-aligned
            to ``candidates``.
        top_k: Number to select.
        lambda_: Relevance/diversity trade-off in ``[0, 1]``.

    Returns:
        The selected candidates in MMR order (original scores preserved).
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    n = len(candidates)
    if n == 0:
        return []
    relevance = _minmax([c.score for c in candidates])
    sims = vectors @ vectors.T  # pairwise cosine similarity

    selected: list[int] = []
    remaining = set(range(n))
    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            def mmr_score(i: int) -> float:
                redundancy = max(float(sims[i, j]) for j in selected)
                return lambda_ * relevance[i] - (1.0 - lambda_) * redundancy

            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.discard(best)
    return [candidates[i] for i in selected]
