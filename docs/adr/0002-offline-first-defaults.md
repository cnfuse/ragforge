# ADR 0002 — Offline-first defaults (local embedder, mock LLM)

- **Status:** Accepted
- **Date:** 2026-06-12

## Context

A RAG system depends on two external capabilities: text embeddings and a language
model. Hard-wiring hosted providers would make the system impossible to run or
test without API keys and network access, slow the test suite, and introduce
non-determinism and cost into CI.

## Decision

Default both dependencies to local, deterministic implementations:

- **`HashingEmbedder`** — feature hashing over token uni/bi-grams into a fixed
  dimensional space, producing L2-normalised vectors. No model download, no
  network, identical output everywhere.
- **`MockLLM`** — drives the exact same tool-use protocol as the Claude client:
  it requests a `search_corpus` call, then answers from the returned passages.

Hosted providers (Anthropic for generation, Voyage for embeddings) are opt-in via
configuration / environment, selected by factory functions (`build_embedder`,
`build_llm`).

## Consequences

- **Positive:** the whole pipeline — including the agent loop — runs and is
  tested with zero network and no API key; CI is fast, free, and deterministic.
- **Positive:** providers are swappable behind `Embedder` / `LLM` protocols.
- **Negative:** the default embedder captures lexical, not deep semantic,
  similarity. This is acceptable for a reference/portfolio system and for
  exercising the pipeline; production deployments select a learned embedder.
