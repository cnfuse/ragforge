# ragforge architecture

This directory documents the architecture of **ragforge** using the
[C4 model](https://c4model.com/) — a hierarchy of diagrams that zoom in from the
system in its environment down to the code that implements it. Each level adds
detail for a different audience.

| Level | Document | Question it answers | Audience |
| ----- | -------- | ------------------- | -------- |
| 1. Context   | [01-context.md](01-context.md)     | How does ragforge fit into the wider world — who uses it and what does it depend on? | Everyone |
| 2. Container | [02-container.md](02-container.md) | What are the separately runnable/deployable pieces and how do they communicate? | Engineers, ops |
| 3. Component | [03-component.md](03-component.md) | What are the major building blocks inside the core, and what are their responsibilities? | Engineers |
| 4. Code      | [04-code.md](04-code.md)           | How are the key abstractions modelled in code? | Contributors |

Architecture decisions are recorded separately as lightweight ADRs in
[`../adr/`](../adr/).

## One-paragraph summary

ragforge is an agentic Retrieval-Augmented Generation platform. Documents are
chunked, embedded, and indexed in both a dense vector store and a sparse BM25
index; at query time a multi-stage retriever fuses dense and sparse rankings
(Reciprocal Rank Fusion) and optionally reranks them, and an agent — driven by
Claude through a tool-use loop — decides when to search and answers with
citations, streaming its progress as events. A built-in evaluation harness scores
both retrieval and answer quality. The same core is exposed through a CLI and an
HTTP API, and runs fully offline by default via a local deterministic embedder
and a mock LLM.

## Quality attributes the design optimises for

- **Testability / hermeticity** — every external dependency (embeddings, LLM)
  sits behind a protocol with a deterministic, offline default, so the whole
  system is exercised in CI with no network or API key.
- **Swappability** — embedders, vector stores, and LLM providers are
  interchangeable behind narrow interfaces; configuration selects the
  implementation.
- **Observability of quality** — retrieval and grounding are measurable, so
  changes are judged against numbers (see the eval harness).
- **Simplicity** — a single-node, in-memory design that is easy to read and
  reason about, with clearly marked seams where it would scale out.
