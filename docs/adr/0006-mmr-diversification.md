# ADR 0006 — MMR diversification of the final selection

- **Status:** Accepted
- **Date:** 2026-06-12

## Context

Ranking purely by relevance frequently fills the top-k with near-duplicate
passages — several chunks from the same section, or repeated boilerplate. For a
RAG context window that wastes budget and starves the model of complementary
evidence, without improving (and sometimes hurting) answer quality.

## Decision

Add an optional **Maximal Marginal Relevance (MMR)** final-selection stage. MMR
greedily builds the result set, each pick maximising

    lambda * relevance(c) - (1 - lambda) * max_sim(c, already_selected)

so it favours passages that are relevant *and* dissimilar to those already
chosen. Relevance is the candidate's incoming score (min-max normalised);
redundancy is cosine similarity between candidate embeddings (dot product, since
vectors are L2-normalised). `lambda = 1` recovers pure relevance ranking.

MMR runs **last**, over the post-fusion / post-rerank candidate pool, and is
off by default (`mmr_enabled`, `mmr_lambda`). The retriever's stage order is now:
dense → hybrid fuse → rerank → **MMR** → top-k.

## Consequences

- **Positive:** diverse, non-redundant context; a recognised technique that
  composes with hybrid retrieval and reranking behind one `retrieve` call.
- **Positive:** fully local/deterministic — re-embeds only the small candidate
  pool, so it stays offline-testable.
- **Negative:** an extra embedding pass over the candidate pool and an O(k·N)
  selection loop. Bounded by `top_k * fetch_multiplier` candidates and gated off
  by default; `lambda` must be tuned per corpus to avoid over-diversifying.
