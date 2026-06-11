# ragforge — build roadmap & progress log

This file is the **source of truth for autonomous build progress**. It is read at
the start of every work session to decide what to do next, and updated at the end
of each session. Each push corresponds to one or more checked items below.

## Vision

A production-shaped, Claude-backed agentic RAG platform with a built-in
evaluation harness — a portfolio piece spanning the full AI-Engineer skill set:
RAG, LLM tool use, evals, API/CLI, testing, CI, and C4 architecture docs.

## Milestones

### M1 — Foundation (retrieval core) ✅
- [x] Project scaffold: `pyproject.toml`, `.gitignore`, `.env.example`, `LICENSE`
- [x] Typed configuration (`config.py`) + structured logging (`logging.py`)
- [x] Domain models (`types.py`): Document, Chunk, ScoredChunk, Answer
- [x] Ingestion: loaders + paragraph-aware chunking with overlap
- [x] Embeddings: `Embedder` protocol + local deterministic `HashingEmbedder`
- [x] Vector store: `InMemoryVectorStore` (cosine, JSON persistence)
- [x] Retriever tying embedder + store
- [x] `Pipeline` assembler from settings
- [x] CLI: `ingest` + `query`
- [x] Tests for chunking, embeddings, store, retrieval

### M2 — LLM & agent layer (Claude)
- [ ] `llm/base.py` — `LLM` protocol + `Message`/`Completion` types
- [ ] `llm/anthropic_client.py` — Claude via Anthropic SDK (adaptive thinking, effort)
- [ ] `llm/mock.py` — deterministic offline LLM for tests
- [ ] `agent/rag_agent.py` — retrieve→prompt→answer with inline citations
- [ ] Tool-use loop: model calls a `search_corpus` tool on demand
- [ ] CLI `ask` subcommand (uses agent; falls back to retrieval if no key)
- [ ] Tests with the mock LLM (no network)

### M3 — Evaluation harness
- [ ] `eval/dataset.py` — QA dataset schema + loader (JSONL)
- [ ] `eval/retrieval_metrics.py` — hit-rate, recall@k, MRR, nDCG
- [ ] `eval/answer_metrics.py` — grounding / citation-faithfulness checks
- [ ] `eval/runner.py` — run a dataset through the pipeline, emit a report
- [ ] CLI `eval` subcommand + sample dataset under `data/`
- [ ] Tests for metrics on synthetic data

### M4 — Service layer
- [ ] `api/app.py` — FastAPI: `/health`, `/ingest`, `/query`, `/ask`
- [ ] Pydantic request/response schemas
- [ ] API tests via `httpx`/`TestClient`
- [ ] `Dockerfile` + container docs

### M5 — Docs & polish
- [ ] C4 docs: context, container, component, code (`docs/architecture/`)
- [ ] Architecture decision records (`docs/adr/`)
- [ ] `CONTRIBUTING.md`, `CHANGELOG.md`, badges in README
- [ ] Voyage embedder implementation (`embeddings/voyage.py`)
- [ ] Expand test coverage; wire coverage gate in CI

## Working agreement (for autonomous sessions)
1. Read this file first; pick the next unchecked item(s).
2. Implement in `D:\playground\ragforge`, using the project `.venv`.
3. Run `pytest` (must stay green) before committing.
4. Make small, coherent commits with clear messages; push to `origin main`.
5. Check off completed items here and append a dated note to the build log.

## Build log
- 2026-06-12 — M1 complete: retrieval core (ingest → embed → store → retrieve),
  CLI, and test suite. Repo initialised and pushed. Next: M2 (Claude agent layer).
