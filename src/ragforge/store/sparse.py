"""A BM25 sparse index over chunk text.

Complements dense embedding retrieval: BM25 captures exact-term importance with
corpus-wide rare-term weighting and document-length normalisation, which dense
vectors approximate only loosely. Fusing the two (see the retriever's RRF) is the
standard hybrid-retrieval recipe. Fully local and deterministic.
"""

from __future__ import annotations

import math
import re

from ragforge.types import Chunk, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """In-memory BM25 ranking over the full chunk corpus."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._docs: list[list[str]] = []
        self._df: dict[str, int] = {}
        self._avgdl = 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> None:
        """Index ``chunks``, updating document-frequency and average length."""
        for chunk in chunks:
            toks = _tokens(chunk.text)
            self._chunks.append(chunk)
            self._docs.append(toks)
            for term in set(toks):
                self._df[term] = self._df.get(term, 0) + 1
        total = sum(len(d) for d in self._docs)
        self._avgdl = total / len(self._docs) if self._docs else 0.0

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Return up to ``top_k`` chunks scored by BM25 (positive scores only)."""
        n = len(self._chunks)
        if n == 0:
            return []
        q_terms = set(_tokens(query))
        scored: list[ScoredChunk] = []
        for chunk, doc in zip(self._chunks, self._docs, strict=True):
            score = self._score(q_terms, doc, n)
            if score > 0:
                scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def _score(self, q_terms: set[str], doc: list[str], n: int) -> float:
        length = len(doc) or 1
        score = 0.0
        for term in q_terms:
            f = doc.count(term)
            if f == 0:
                continue
            df = self._df.get(term, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            denom = f + self.k1 * (1 - self.b + self.b * length / (self._avgdl or 1.0))
            score += idf * (f * (self.k1 + 1)) / denom
        return score
