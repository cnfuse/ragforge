# ADR 0003 — Agentic tool use over context stuffing

- **Status:** Accepted
- **Date:** 2026-06-12

## Context

A RAG answer can be produced two ways:

1. **Context stuffing** — always retrieve top-k chunks and prepend them to the
   prompt, then ask for an answer.
2. **Agentic tool use** — expose a search tool and let the model decide whether,
   when, and what to search, possibly across multiple turns.

Stuffing is simpler but always pays retrieval cost, can flood the prompt with
irrelevant context, and can't reformulate a query when the first search is weak.

## Decision

Implement the agent as a bounded **tool-use loop** (`RagAgent`) that exposes a
single `search_corpus` tool. The model issues searches on demand; the agent
executes them against the `Pipeline`, feeds results back as `tool_result`
messages, and accumulates every retrieved chunk as evidence. The loop is capped
by `max_steps` to bound cost and guarantee termination.

The loop is provider-agnostic: assistant turns and tool results are reconstructed
in the Anthropic content-block shape, so the identical loop runs against the live
Claude client and the offline mock.

## Consequences

- **Positive:** the model can skip retrieval when unnecessary, search with a
  refined query, and ground its answer in exactly the evidence it pulled.
- **Positive:** citations reflect the union of all searches, scored and sorted.
- **Negative:** more moving parts than stuffing, and answer quality depends on
  the model's tool-use behaviour. Mitigated by the `max_steps` guard and by the
  evaluation harness, which measures grounding and correctness.
