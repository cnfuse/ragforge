"""High-level assembly of a RAG pipeline from configuration.

This is the single entry point most callers want: hand it :class:`Settings`
and it wires together an embedder, a vector store, and a retriever consistent
with that configuration. Higher layers (CLI, API, agent) build on top of the
:class:`Pipeline` returned here rather than constructing components by hand.
"""

from __future__ import annotations

from pathlib import Path

from ragforge.config import Settings
from ragforge.config import settings as default_settings
from ragforge.embeddings import build_embedder
from ragforge.ingestion.chunking import chunk_document
from ragforge.logging import get_logger
from ragforge.rerank import build_reranker
from ragforge.retrieval.retriever import Retriever
from ragforge.store.memory import InMemoryVectorStore
from ragforge.types import Document, ScoredChunk

log = get_logger("pipeline")


class Pipeline:
    """A configured ingest-and-retrieve pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self.embedder = build_embedder(self.settings)
        self.store = InMemoryVectorStore(dim=self.embedder.dim)
        self.retriever = self._build_retriever(self.store)

    def _build_retriever(self, store: InMemoryVectorStore) -> Retriever:
        return Retriever(
            self.embedder,
            store,
            reranker=build_reranker(self.settings),
            fetch_multiplier=self.settings.rerank_fetch_multiplier,
        )

    @classmethod
    def from_index(cls, path: str | Path, settings: Settings | None = None) -> Pipeline:
        """Build a pipeline whose store is loaded from a saved index file.

        The configured embedder must match the index's dimensionality, since a
        query is only comparable to vectors produced by the same embedder.
        """
        pipeline = cls(settings)
        store = InMemoryVectorStore.load(path)
        if store.dim != pipeline.embedder.dim:
            raise ValueError(
                f"index dim {store.dim} != embedder dim {pipeline.embedder.dim}; "
                "re-ingest with the current embedding configuration"
            )
        pipeline.store = store
        pipeline.retriever = pipeline._build_retriever(store)
        return pipeline

    def ingest(self, documents: list[Document]) -> int:
        """Chunk and index documents; returns the number of chunks indexed."""
        chunks = []
        for doc in documents:
            chunks.extend(
                chunk_document(
                    doc,
                    chunk_size=self.settings.chunk_size,
                    overlap=self.settings.chunk_overlap,
                )
            )
        return self.retriever.index(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        """Retrieve the most relevant chunks for ``query``."""
        return self.retriever.retrieve(query, top_k=top_k or self.settings.top_k)
