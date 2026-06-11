# C4 Level 3 — Components

This zooms into the **Core library** container and shows its major components
(Python modules/packages) and their dependencies. Arrows point from a component
to the components it uses.

```mermaid
C4Component
    title Component Diagram — ragforge core

    Container_Boundary(core, "Core library (ragforge)") {
        Component(config, "config / logging / types", "pydantic-settings", "Typed settings, structured logging, and shared domain models (Document, Chunk, ScoredChunk, Answer).")
        Component(ingestion, "ingestion", "loaders + chunking", "Loads files/dirs into Documents; paragraph-aware chunking with overlap.")
        Component(embeddings, "embeddings", "Embedder protocol", "HashingEmbedder (local default) and optional Voyage provider; build_embedder factory.")
        Component(store, "store", "VectorStore protocol", "InMemoryVectorStore — cosine search over a NumPy matrix, JSON persistence.")
        Component(retrieval, "retrieval", "Retriever", "Embed-and-index on ingest; embed-and-search on query.")
        Component(pipeline, "pipeline", "Pipeline", "Assembles embedder + store + retriever from settings; ingest/retrieve entry point.")
        Component(llm, "llm", "LLM protocol", "AnthropicLLM (Claude) and MockLLM (offline); build_llm factory.")
        Component(agent, "agent", "RagAgent", "Bounded tool-use loop exposing search_corpus; answers with citations.")
        Component(eval, "eval", "harness", "Dataset, IR metrics, answer grounding, runner, typed report.")
    }

    System_Ext(anthropic, "Anthropic API", "Claude")
    System_Ext(voyage, "Voyage AI", "Embeddings")

    Rel(ingestion, config, "uses types")
    Rel(embeddings, voyage, "embeds (optional)", "HTTPS")
    Rel(retrieval, embeddings, "embeds via")
    Rel(retrieval, store, "searches")
    Rel(pipeline, ingestion, "chunks")
    Rel(pipeline, retrieval, "delegates")
    Rel(pipeline, embeddings, "builds")
    Rel(pipeline, store, "owns")
    Rel(llm, anthropic, "calls (optional)", "HTTPS")
    Rel(agent, pipeline, "retrieves via")
    Rel(agent, llm, "generates via")
    Rel(eval, pipeline, "scores retrieval")
    Rel(eval, agent, "scores answers")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

## Component responsibilities

| Component | Module(s) | Responsibility |
| --------- | --------- | -------------- |
| **Config / logging / types** | `config.py`, `logging.py`, `types.py` | One typed view of configuration; consistent logging; the domain models (`Document`, `Chunk`, `ScoredChunk`, `Answer`) that flow between components. |
| **Ingestion** | `ingestion/loaders.py`, `ingestion/chunking.py` | Turn raw text/files into `Document`s; split into overlapping, paragraph-aware `Chunk`s. |
| **Embeddings** | `embeddings/base.py`, `hashing.py`, `__init__.build_embedder` | `Embedder` protocol; local deterministic `HashingEmbedder`; optional Voyage provider, chosen by config. |
| **Store** | `store/base.py`, `store/memory.py` | `VectorStore` protocol; `InMemoryVectorStore` doing cosine search over a NumPy matrix with JSON persistence. |
| **Retrieval** | `retrieval/retriever.py` | Bind an embedder to a store: embed-and-index, embed-and-search. |
| **Pipeline** | `pipeline.py` | Assemble components from `Settings`; the high-level ingest/retrieve facade (`Pipeline`, `Pipeline.from_index`). |
| **LLM** | `llm/base.py`, `anthropic_client.py`, `mock.py`, `build_llm` | Provider-agnostic `LLM` contract; Claude adapter; deterministic offline mock; factory that picks live-vs-mock by key presence. |
| **Agent** | `agent/rag_agent.py` | The agentic loop: expose `search_corpus`, let the model decide when to search, collect evidence, answer with citations. |
| **Eval** | `eval/*` | Dataset format, retrieval metrics, answer grounding, runner, typed report. |

## Key design seams

- **`Embedder`** and **`VectorStore`** are `Protocol`s — the retriever and
  pipeline depend on the interface, not a concrete class. Swapping providers is a
  config change, not a code change.
- **`LLM`** is likewise a protocol; `AnthropicLLM` and `MockLLM` are
  interchangeable, which is what makes the agent loop testable offline.
- The **agent depends on the `Pipeline`** for retrieval and on an `LLM` for
  generation — it never touches embedders or stores directly.
