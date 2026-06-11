# ADR 0004 — Two-stage retrieval with a hybrid reranker

- **Status:** Accepted
- **Date:** 2026-06-12

## Context

Single-stage embedding retrieval optimises recall and is cheap, but pure vector
similarity can rank a semantically-close-but-wrong passage above one that
contains the query's exact salient terms. The default local `HashingEmbedder`
captures lexical signal only coarsely, so precision at the top of the list
suffers on term-specific queries.

## Decision

Add an optional **second stage** behind a `Reranker` protocol. When enabled, the
store returns `top_k * fetch_multiplier` recall-oriented candidates, and the
reranker reorders them down to `top_k`.

The default `LexicalReranker` is a **hybrid**: it computes a BM25 score over the
candidate set (rare-term weighting + length normalisation) and blends it,
min-max-normalised, with the first-pass embedding score via a tunable `alpha`
(weight on embeddings vs. BM25). It is local and deterministic.

A second implementation, `LLMReranker`, asks the language model to judge each
candidate's relevance in a single structured-JSON call (Claude-as-judge),
falling back to first-pass order if the output can't be parsed. Both satisfy the
same `Reranker` protocol and are selected by `rerank_provider`
(`lexical` | `llm`).

Reranking is **off by default** (`rerank_enabled=False`) so existing behaviour
and the single-stage tests are unchanged; it is opt-in via configuration.

## Consequences

- **Positive:** term-exact matches are promoted without discarding semantic
  ranking; `alpha` tunes the balance. The seam (`Reranker` protocol) cleanly
  admits a future cross-encoder or LLM reranker.
- **Positive:** two-stage retrieval is the standard production RAG pattern, so
  the design mirrors real systems.
- **Negative:** a second stage adds latency proportional to the candidate count
  and a parameter (`alpha`) to tune. Mitigated by keeping it optional and by the
  evaluation harness, which quantifies the precision change.
