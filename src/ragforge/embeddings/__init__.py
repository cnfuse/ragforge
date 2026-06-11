"""Embedding providers.

The default :class:`HashingEmbedder` is fully local and deterministic, so the
whole pipeline (and its tests) runs with no network access or API key. A
Voyage-backed provider can be selected via configuration for production-quality
semantic embeddings.
"""

from __future__ import annotations

from ragforge.config import Settings
from ragforge.embeddings.base import Embedder
from ragforge.embeddings.hashing import HashingEmbedder

__all__ = ["Embedder", "HashingEmbedder", "build_embedder"]


def build_embedder(settings: Settings) -> Embedder:
    """Construct the embedder named by ``settings.embedding_provider``."""
    provider = settings.embedding_provider.lower()
    if provider == "hashing":
        return HashingEmbedder(dim=settings.embedding_dim)
    if provider == "voyage":
        from ragforge.embeddings.voyage import VoyageEmbedder

        return VoyageEmbedder(model=settings.voyage_model)
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider!r}")
