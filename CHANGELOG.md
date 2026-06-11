# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Retrieval core (M1):** typed config, structured logging, domain models;
  file/directory loaders; paragraph-aware chunking with overlap; local
  deterministic `HashingEmbedder`; `InMemoryVectorStore` with cosine search and
  JSON persistence; `Retriever` and `Pipeline`; `ingest`/`query` CLI.
- **Claude agent layer (M2):** provider-agnostic `LLM` protocol; `AnthropicLLM`
  (adaptive thinking + effort); deterministic offline `MockLLM`; `RagAgent`
  tool-use loop with `search_corpus` and citations; `ask` CLI;
  `Pipeline.from_index`.
- **Evaluation harness (M3):** JSONL QA dataset; retrieval metrics
  (hit-rate, recall@k, MRR, nDCG); answer expected-match + grounding; typed
  `EvalReport`; runner; `eval` CLI; sample corpus + dataset.
- **HTTP service (M4):** FastAPI app (`/health`, `/ingest`, `/query`, `/ask`)
  with pydantic schemas and OpenAPI docs; multi-stage Dockerfile.
- **Documentation (M5):** full C4 architecture model (context, container,
  component, code) with Mermaid diagrams; ADRs; `CONTRIBUTING`; optional Voyage
  embedder.
- **Reranking (M6):** optional two-stage retrieval — a `Reranker` protocol with
  a hybrid BM25 + embedding `LexicalReranker` and an `LLMReranker`
  (Claude-as-judge); config-driven and off by default (ADR 0004). Strict,
  blocking `mypy` in CI.
- **Persistence & DX (M6):** durable API index (`RAGFORGE_INDEX_PATH`,
  load-on-startup, auto-save, `/save`); runnable `examples/quickstart.py`.

[Unreleased]: https://github.com/cnfuse/ragforge/commits/main
