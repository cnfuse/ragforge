# C4 Level 4 — Code

The lowest C4 level shows how the key abstractions are modelled in code. Two
views are most illustrative: the **class relationships** around the agent, and
the **sequence** of the tool-use loop.

## Class diagram — core abstractions

```mermaid
classDiagram
    class Document {
        +str id
        +str text
        +dict metadata
    }
    class Chunk {
        +str id
        +str doc_id
        +str text
        +int ordinal
    }
    class ScoredChunk {
        +Chunk chunk
        +float score
    }
    class Answer {
        +str question
        +str text
        +list~ScoredChunk~ citations
        +str model
    }

    class Embedder {
        <<protocol>>
        +int dim
        +embed(texts) ndarray
        +embed_one(text) ndarray
    }
    class HashingEmbedder
    class VoyageEmbedder

    class VectorStore {
        <<protocol>>
        +add(chunks, vectors)
        +search(vec, top_k) list~ScoredChunk~
    }
    class InMemoryVectorStore
    class BM25Index {
        +add(chunks)
        +search(query, top_k) list~ScoredChunk~
    }

    class Reranker {
        <<protocol>>
        +rerank(query, candidates, top_k) list~ScoredChunk~
    }
    class LexicalReranker
    class LLMReranker

    class Retriever {
        +index(chunks) int
        +retrieve(query, top_k) list~ScoredChunk~
        -_rrf_fuse(dense, sparse, top_k)
    }
    class Pipeline {
        +Settings settings
        +ingest(docs) int
        +retrieve(query, top_k) list~ScoredChunk~
        +from_index(path) Pipeline
    }

    class LLM {
        <<protocol>>
        +generate(system, messages, tools, max_tokens) LLMResponse
    }
    class AnthropicLLM
    class MockLLM

    class RagAgent {
        +int max_steps
        +answer(question) Answer
        +iter_events(question) Iterator~AgentEvent~
    }
    class AgentEvent {
        +str type
        +int step
        +Answer answer
    }

    Embedder <|.. HashingEmbedder
    Embedder <|.. VoyageEmbedder
    VectorStore <|.. InMemoryVectorStore
    Reranker <|.. LexicalReranker
    Reranker <|.. LLMReranker
    LLM <|.. AnthropicLLM
    LLM <|.. MockLLM

    Retriever --> Embedder
    Retriever --> VectorStore
    Retriever --> BM25Index : hybrid (optional)
    Retriever --> Reranker : rerank (optional)
    LLMReranker --> LLM
    Pipeline --> Retriever
    RagAgent --> Pipeline
    RagAgent --> LLM
    RagAgent ..> AgentEvent : yields
    AgentEvent o-- Answer
    Answer o-- ScoredChunk
    ScoredChunk o-- Chunk
```

## Sequence diagram — the agentic tool-use loop

`RagAgent.iter_events()` runs a bounded loop and *yields* progress events; the
model is offered a `search_corpus` tool and decides when to use it. The agent
executes searches (each a full dense→hybrid→rerank retrieve), feeds results back,
and collects every retrieved chunk as evidence. `answer()` simply drains this
generator and returns the terminal event's `Answer`; the SSE endpoint and
`ask --stream` forward the events themselves.

```mermaid
sequenceDiagram
    actor User
    participant Agent as RagAgent
    participant LLM
    participant Pipeline
    participant Retriever

    User->>Agent: iter_events("which language is memory safe?")
    loop until end_turn or max_steps
        Agent->>LLM: generate(system, messages, tools=[search_corpus])
        alt model requests a tool
            LLM-->>Agent: stop_reason=tool_use, ToolCall(search_corpus, query)
            Agent-->>User: yield AgentEvent(type=search)
            Agent->>Pipeline: retrieve(query, top_k)
            Pipeline->>Retriever: dense -> hybrid fuse -> rerank
            Retriever-->>Pipeline: ranked ScoredChunks
            Pipeline-->>Agent: ScoredChunks
            Agent-->>User: yield AgentEvent(type=results, count=n)
            Note over Agent: record chunks as citations<br/>append tool_result to messages
        else model answers
            LLM-->>Agent: stop_reason=end_turn, text
            Agent-->>User: yield AgentEvent(type=answer, Answer)
        end
    end
```

## Why these shapes

- **Protocols over base classes.** `Embedder`, `VectorStore`, `Reranker`, and
  `LLM` are `typing.Protocol`s. Concrete types are structurally compatible without
  inheritance, which keeps providers decoupled and trivially mockable.
- **Retrieval is staged, not branched at call sites.** `Retriever.retrieve`
  internally runs dense search, optional RRF fusion with the `BM25Index`, and an
  optional `Reranker` — callers (agent, CLI, API) see one method whatever the
  configured depth.
- **One streaming loop, three consumers.** `iter_events` is the single loop;
  `answer()`, the SSE endpoint, and `ask --stream` are all thin consumers of it,
  so behaviour can't drift between them.
- **`LLMResponse.wants_tools`** centralises the "should I run tools?" decision
  (`stop_reason == "tool_use"` and at least one call), so the agent loop reads
  declaratively.
- **Reconstructed assistant turns.** The agent rebuilds the assistant message
  (text + `tool_use` blocks) and the following `tool_result` user message in the
  Anthropic content-block shape, so the *same* loop drives both the live client
  and the offline mock with no special-casing.
- **Evidence is accumulated, not just last-search.** Chunks from every search in
  the loop are merged (keeping the best score per chunk) and returned sorted, so
  the `Answer.citations` reflect everything the answer could draw on.
