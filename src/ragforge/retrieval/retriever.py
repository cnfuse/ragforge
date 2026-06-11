"""The retriever ties an embedder to a vector store.

It owns the two operations that matter to the rest of the system: indexing
chunks (embed then store) and answering a query (embed then search). Keeping
both behind one object means the agent never has to know which embedder or store
is in play.
"""

from __future__ import annotations

from ragforge.embeddings.base import Embedder
from ragforge.logging import get_logger
from ragforge.store.base import VectorStore
from ragforge.types import Chunk, ScoredChunk

log = get_logger("retrieval")


class Retriever:
    """Embed-and-store on ingest; embed-and-search on query."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def index(self, chunks: list[Chunk]) -> int:
        """Embed and add ``chunks`` to the store, returning the count indexed."""
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        log.info("indexed %d chunks (store now holds %d)", len(chunks), len(self.store))
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """Return the ``top_k`` chunks most relevant to ``query``."""
        if not query.strip():
            return []
        query_vector = self.embedder.embed_one(query)
        return self.store.search(query_vector, top_k=top_k)
