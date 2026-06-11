# C4 Level 1 — System Context

The context diagram shows ragforge as a single black box, the people who use it,
and the external systems it depends on.

```mermaid
C4Context
    title System Context — ragforge

    Person(engineer, "AI Engineer / Developer", "Indexes a corpus and asks questions over it; evaluates retrieval and answer quality.")
    Person(enduser, "Application / End user", "Calls the HTTP API to get grounded answers from the corpus.")

    System(ragforge, "ragforge", "Agentic RAG platform with a built-in evaluation harness. Ingests documents, retrieves relevant context, and answers questions with citations.")

    System_Ext(anthropic, "Anthropic API", "Claude models — generate answers and drive the tool-use loop (optional; offline mock used when absent).")
    System_Ext(voyage, "Voyage AI", "Semantic embedding model (optional; local hashing embedder used by default).")

    Rel(engineer, ragforge, "Ingests docs, queries, asks, evaluates", "CLI")
    Rel(enduser, ragforge, "Submits documents and questions", "HTTPS / JSON")
    Rel(ragforge, anthropic, "Generates answers, requests tool calls", "HTTPS (Messages API)")
    Rel(ragforge, voyage, "Embeds text", "HTTPS")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

## Actors

- **AI Engineer / Developer** — the primary user. Builds an index from a corpus,
  runs queries and agent-backed `ask` sessions, and uses the evaluation harness
  to measure quality while iterating.
- **Application / End user** — any client that consumes the HTTP API to ingest
  documents and ask questions, embedding ragforge into a larger product.

## External dependencies

- **Anthropic API (optional).** When `ANTHROPIC_API_KEY` is set, the agent calls
  Claude (default `claude-opus-4-8`, adaptive thinking + effort) to answer and to
  decide when to invoke the `search_corpus` tool. Without a key, a deterministic
  offline mock drives the identical loop, so the system is fully functional and
  testable with no third-party calls.
- **Voyage AI (optional).** Selectable as the embedding provider for
  production-quality semantic vectors. The default is a local, deterministic
  hashing embedder that requires no network or key.

## Boundaries & trust

The only data leaving the process is (a) text sent to Anthropic for answer
generation and (b) text sent to Voyage for embedding — both only when explicitly
configured. API keys are read from the environment by the respective SDKs and are
never logged or persisted by ragforge.
