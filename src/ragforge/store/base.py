"""Vector store protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ragforge.types import Chunk, ScoredChunk


class VectorStore(Protocol):
    """Stores chunk embeddings and answers nearest-neighbour queries."""

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Index ``chunks`` alongside their pre-computed ``vectors``."""
        ...

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        """Return the ``top_k`` chunks most similar to ``query_vector``."""
        ...

    def __len__(self) -> int:  # noqa: D105
        ...
