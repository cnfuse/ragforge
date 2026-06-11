# ADR 0005 — Hybrid dense+sparse retrieval with Reciprocal Rank Fusion

- **Status:** Accepted
- **Date:** 2026-06-12

## Context

Dense embedding retrieval captures semantic similarity but can miss documents
that match a query's exact rare terms (names, identifiers, jargon) — especially
with the local hashing embedder. Sparse lexical retrieval (BM25) is the
complementary failure mode: great on exact terms, blind to paraphrase. Production
RAG systems routinely combine both.

## Decision

Add an optional **sparse `BM25Index`** over chunk text and fuse it with dense
retrieval using **Reciprocal Rank Fusion (RRF)**: each document's fused score is
the sum over rankings of `1 / (rrf_k + rank)`. RRF is parameter-light (one
constant, `rrf_k`, default 60), needs no score normalisation across
incomparable scales, and is robust — the standard choice for combining rankings.

The retriever now composes up to three optional stages: **dense → hybrid fusion →
rerank**. Hybrid is off by default (`hybrid_enabled`); when on, the sparse index
is built during ingest and **rebuilt from the persisted chunks** when an index is
loaded from disk, so it survives the save/load cycle.

## Consequences

- **Positive:** recall improves on term-specific queries without losing semantic
  matches; composes cleanly with the existing reranker (fusion produces the
  candidate set the reranker reorders).
- **Positive:** RRF avoids the fragile score-normalisation that naive
  weighted-sum fusion requires.
- **Negative:** a second in-memory index roughly doubles index memory and adds a
  BM25 pass per query. Mitigated by keeping it optional and bounding fusion to
  `top_k * fetch_multiplier` candidates.
