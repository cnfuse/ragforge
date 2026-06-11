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

    class Retriever {
        +index(chunks) int
        +retrieve(query, top_k) list~ScoredChunk~
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
    class LLMResponse {
        +str text
        +list~ToolCall~ tool_calls
        +str stop_reason
        +bool wants_tools
    }

    class RagAgent {
        +int max_steps
        +answer(question) Answer
    }

    Embedder <|.. HashingEmbedder
    Embedder <|.. VoyageEmbedder
    VectorStore <|.. InMemoryVectorStore
    LLM <|.. AnthropicLLM
    LLM <|.. MockLLM

    Retriever --> Embedder
    Retriever --> VectorStore
    Pipeline --> Retriever
    Pipeline --> Embedder
    Pipeline --> VectorStore
    RagAgent --> Pipeline
    RagAgent --> LLM
    RagAgent ..> Answer : produces
    Answer o-- ScoredChunk
    ScoredChunk o-- Chunk
```

## Sequence diagram — the agentic tool-use loop

`RagAgent.answer()` runs a bounded loop. The model is offered a `search_corpus`
tool and decides when to use it; the agent executes searches, feeds results back,
and collects every retrieved chunk as evidence for the final answer.

```mermaid
sequenceDiagram
    actor User
    participant Agent as RagAgent
    participant LLM
    participant Pipeline
    participant Store as VectorStore

    User->>Agent: answer("which language is memory safe?")
    loop until end_turn or max_steps
        Agent->>LLM: generate(system, messages, tools=[search_corpus])
        alt model requests a tool
            LLM-->>Agent: stop_reason=tool_use, ToolCall(search_corpus, query)
            Agent->>Pipeline: retrieve(query, top_k)
            Pipeline->>Store: search(query_vector, top_k)
            Store-->>Pipeline: ranked ScoredChunks
            Pipeline-->>Agent: ScoredChunks
            Note over Agent: record chunks as citations<br/>append tool_result to messages
        else model answers
            LLM-->>Agent: stop_reason=end_turn, text
            Agent-->>User: Answer(text, citations sorted by score)
        end
    end
```

## Why these shapes

- **Protocols over base classes.** `Embedder`, `VectorStore`, and `LLM` are
  `typing.Protocol`s. Concrete types are structurally compatible without
  inheritance, which keeps providers decoupled and trivially mockable.
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
