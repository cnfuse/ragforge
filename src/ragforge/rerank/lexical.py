"""Hybrid lexical reranker: BM25 over the candidate set, blended with embeddings.

The first-pass embedding score captures semantic similarity; BM25 captures exact
term importance (with length normalisation and rare-term weighting). Blending the
two min-max-normalised signals gives a hybrid score that promotes candidates
matching the query's salient words without discarding semantic ranking. It is
fully local and deterministic.
"""

from __future__ import annotations

import math
import re

from ragforge.types import ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _minmax(values: list[float]) -> list[float]:
    """Scale values to [0, 1]; a flat list maps to all-1.0 (no signal to rank on)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class LexicalReranker:
    """BM25 + embedding hybrid reranker over a small candidate set."""

    def __init__(self, alpha: float = 0.5, *, k1: float = 1.5, b: float = 0.75) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha  # weight on the embedding score; (1-alpha) on BM25
        self.k1 = k1
        self.b = b

    def _bm25(self, query: str, docs: list[list[str]]) -> list[float]:
        n = len(docs)
        if n == 0:
            return []
        avgdl = sum(len(d) for d in docs) / n or 1.0
        # Document frequency of each query term within the candidate set.
        q_terms = set(_tokens(query))
        df = {t: sum(1 for d in docs if t in d) for t in q_terms}
        scores: list[float] = []
        for doc in docs:
            length = len(doc) or 1
            score = 0.0
            for t in q_terms:
                f = doc.count(t)
                if f == 0:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * length / avgdl)
                score += idf * (f * (self.k1 + 1)) / denom
            scores.append(score)
        return scores

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        docs = [_tokens(c.chunk.text) for c in candidates]
        bm25 = _minmax(self._bm25(query, docs))
        embed = _minmax([c.score for c in candidates])

        blended = [
            ScoredChunk(
                chunk=cand.chunk,
                score=self.alpha * e + (1.0 - self.alpha) * b,
            )
            for cand, e, b in zip(candidates, embed, bm25, strict=True)
        ]
        blended.sort(key=lambda s: s.score, reverse=True)
        return blended[:top_k]
