"""The retriever ties an embedder to a vector store.

It owns the two operations that matter to the rest of the system: indexing
chunks (embed then store) and answering a query (embed then search). Keeping
both behind one object means the agent never has to know which embedder or store
is in play.
"""

from __future__ import annotations

from ragforge.embeddings.base import Embedder
from ragforge.logging import get_logger
from ragforge.rerank.base import Reranker
from ragforge.store.base import VectorStore
from ragforge.store.sparse import BM25Index
from ragforge.types import Chunk, ScoredChunk

log = get_logger("retrieval")


class Retriever:
    """Embed-and-store on ingest; multi-stage search on query.

    Up to three stages compose, each optional and config-gated:

    1. **Dense** embedding search over the vector store (always).
    2. **Hybrid** fusion — if a :class:`BM25Index` is supplied, dense and sparse
       rankings are merged with Reciprocal Rank Fusion (RRF).
    3. **Rerank** — if a :class:`~ragforge.rerank.base.Reranker` is supplied, the
       fused candidates are reordered down to ``top_k``.

    When stage 2 or 3 is active the store is queried for
    ``top_k * fetch_multiplier`` recall-oriented candidates first.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        reranker: Reranker | None = None,
        sparse: BM25Index | None = None,
        fetch_multiplier: int = 4,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.sparse = sparse
        self.fetch_multiplier = max(1, fetch_multiplier)
        self.rrf_k = rrf_k

    def index(self, chunks: list[Chunk]) -> int:
        """Embed and add ``chunks`` to the dense store (and sparse index if any)."""
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        if self.sparse is not None:
            self.sparse.add(chunks)
        log.info("indexed %d chunks (store now holds %d)", len(chunks), len(self.store))
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """Return the ``top_k`` chunks most relevant to ``query``."""
        if not query.strip():
            return []
        multi_stage = self.sparse is not None or self.reranker is not None
        fetch_n = top_k * self.fetch_multiplier if multi_stage else top_k

        query_vector = self.embedder.embed_one(query)
        dense = self.store.search(query_vector, top_k=fetch_n)

        if self.sparse is not None:
            sparse = self.sparse.search(query, top_k=fetch_n)
            candidates = self._rrf_fuse(dense, sparse, top_k=fetch_n)
        else:
            candidates = dense

        if self.reranker is not None:
            return self.reranker.rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]

    def _rrf_fuse(
        self, dense: list[ScoredChunk], sparse: list[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]:
        """Reciprocal Rank Fusion: combine two rankings by summed reciprocal rank."""
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for ranking in (dense, sparse):
            for rank, sc in enumerate(ranking):
                chunks[sc.chunk.id] = sc.chunk
                scores[sc.chunk.id] = scores.get(sc.chunk.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        merged = [ScoredChunk(chunk=chunks[cid], score=score) for cid, score in scores.items()]
        merged.sort(key=lambda s: s.score, reverse=True)
        return merged[:top_k]
