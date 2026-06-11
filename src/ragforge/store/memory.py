"""An in-memory vector store backed by a single NumPy matrix.

Embeddings are stored as rows of a contiguous ``(n, dim)`` matrix, so a query is
one matrix-vector product followed by a partial sort. Because vectors are
L2-normalised upstream, the dot product is cosine similarity. This scales
comfortably to tens of thousands of chunks — ample for a single-node service —
and the on-disk format is plain JSON for easy inspection and portability.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ragforge.types import Chunk, ScoredChunk


class InMemoryVectorStore:
    """Brute-force cosine-similarity store with JSON persistence."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._chunks: list[Chunk] = []
        self._matrix = np.empty((0, dim), dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        """The indexed chunks, in insertion order (read-only view)."""
        return list(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors must have the same length")
        if vectors.size and vectors.shape[1] != self._dim:
            raise ValueError(f"expected dim {self._dim}, got {vectors.shape[1]}")
        if not chunks:
            return
        self._chunks.extend(chunks)
        self._matrix = np.vstack([self._matrix, vectors.astype(np.float32)])

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if len(self._chunks) == 0:
            return []
        k = min(top_k, len(self._chunks))
        scores = self._matrix @ query_vector.astype(np.float32)
        # argpartition gives the top-k cheaply, then we sort just those k.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [ScoredChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top]

    # --- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Write the index to a single JSON file."""
        payload = {
            "dim": self._dim,
            "chunks": [c.model_dump() for c in self._chunks],
            "vectors": self._matrix.tolist(),
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> InMemoryVectorStore:
        """Load an index previously written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(dim=int(payload["dim"]))
        chunks = [Chunk(**c) for c in payload["chunks"]]
        vectors = np.array(payload["vectors"], dtype=np.float32).reshape(-1, store.dim)
        store.add(chunks, vectors)
        return store
