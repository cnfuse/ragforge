"""Embedder protocol shared by every provider."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Maps text to dense, L2-normalised vectors.

    Implementations must return an array of shape ``(len(texts), dim)`` with
    rows already normalised, so a dot product equals cosine similarity.
    """

    @property
    def dim(self) -> int:  # noqa: D102
        ...

    def embed(self, texts: list[str]) -> np.ndarray:  # noqa: D102
        ...

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string and return a 1-D vector of length ``dim``."""
        vector: np.ndarray = self.embed([text])[0]
        return vector
