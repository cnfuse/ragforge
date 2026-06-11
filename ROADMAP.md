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

### M2 — LLM & agent layer (Claude) ✅
- [x] `llm/base.py` — `LLM` protocol + `ToolSpec`/`ToolCall`/`LLMResponse` types
- [x] `llm/anthropic_client.py` — Claude via Anthropic SDK (adaptive thinking, effort)
- [x] `llm/mock.py` — deterministic offline LLM for tests
- [x] `agent/rag_agent.py` — bounded tool-use loop, answer with citations
- [x] Tool-use loop: model calls a `search_corpus` tool on demand
- [x] CLI `ask` subcommand (Claude when key set, else offline mock)
- [x] `Pipeline.from_index` to load a saved index for answering
- [x] Tests with the mock LLM (no network) — 30 tests total

### M3 — Evaluation harness ✅
- [x] `eval/dataset.py` — QA dataset schema + JSONL loader
- [x] `eval/retrieval_metrics.py` — hit-rate, recall@k, MRR, nDCG
- [x] `eval/answer_metrics.py` — expected-match + grounding/faithfulness checks
- [x] `eval/report.py` — typed `EvalReport` (aggregates + per-query rows)
- [x] `eval/runner.py` — retrieval + answer evaluation over a dataset
- [x] CLI `eval` subcommand + sample corpus & dataset under `data/sample/`
- [x] Tests for metrics + runner on synthetic data (40 tests total)

### M4 — Service layer ✅
- [x] `api/app.py` — FastAPI factory: `/health`, `/ingest`, `/query`, `/ask`
- [x] Pydantic request/response schemas (`api/schemas.py`)
- [x] API tests via `TestClient` (7 tests, incl. validation + OpenAPI)
- [x] Multi-stage `Dockerfile` (non-root, healthcheck) + `.dockerignore`
- [x] README: API + Docker + eval usage documented

### M5 — Docs & polish ✅
- [x] C4 docs: context, container, component, code (`docs/architecture/`) w/ Mermaid
- [x] Architecture decision records (`docs/adr/` — 3 ADRs)
- [x] `CONTRIBUTING.md`, `CHANGELOG.md`, badges in README
- [x] Voyage embedder implementation (`embeddings/voyage.py`)
- [x] Expanded tests (factory/config selection); coverage reported in CI

## Planned roadmap complete (M1–M5) 🎉
The originally scoped platform is built. Future autonomous sessions harden and
extend it — candidate work below.

### M6 — Hardening & extensions (backlog)
- [x] Persisted index reuse in the API — `RAGFORGE_INDEX_PATH` load-on-startup,
      auto-save after ingest, `/save` endpoint
- [x] Reranking stage behind a `Reranker` protocol — hybrid BM25+embedding
      `LexicalReranker`, optional two-stage retrieval, config-driven
- [x] Tighten mypy to fully strict and make it blocking in CI
- [x] `examples/quickstart.py` — runnable end-to-end demo (ingest→query→ask→eval)
- [x] Optional LLM reranker (`LLMReranker`, Claude-as-judge) on the same protocol;
      selected via `rerank_provider`
- [x] Streaming `/ask/stream` (SSE) — agent emits search/results/answer events
- [x] Coverage gate — `fail_under=90` (network adapters omitted); CLI now tested
- [x] CLI streaming output — `ask --stream` consumes the agent event stream
- [x] Hybrid dense+sparse first-pass retrieval — BM25 index fused with dense
      via Reciprocal Rank Fusion (ADR 0005); composes with rerank
- [ ] Cross-encoder reranker variant
- [ ] Persist the sparse index alongside the dense index (currently rebuilt)

## Working agreement (for autonomous sessions)
1. Read this file first; pick the next unchecked item(s).
2. Implement in `D:\playground\ragforge`, using the project `.venv`.
3. Run `pytest` (must stay green) before committing.
4. Make small, coherent commits with clear messages; push to `origin main`.
5. Check off completed items here and append a dated note to the build log.

## Build log
- 2026-06-12 — M1 complete: retrieval core (ingest → embed → store → retrieve),
  CLI, and test suite. Repo initialised and pushed. Next: M2 (Claude agent layer).
- 2026-06-12 — M2 complete: provider-agnostic LLM contract, Claude client
  (adaptive thinking + effort), deterministic offline MockLLM, agentic
  search-and-answer loop with citations, `ask` CLI, `Pipeline.from_index`.
  30 tests green, lint clean. Next: M3 (evaluation harness).
- 2026-06-12 — M3 complete: evaluation harness — JSONL dataset format, IR
  metrics (hit-rate/recall@k/MRR/nDCG), answer expected-match + grounding,
  typed report, runner, and `eval` CLI with a sample corpus/dataset. On the
  sample set: retrieval 1.000 across the board, grounding 0.887. 40 tests
  green, lint clean. Next: M4 (FastAPI service).
- 2026-06-12 — M4 complete: FastAPI service (/health, /ingest, /query, /ask)
  with pydantic schemas and OpenAPI docs, TestClient suite, multi-stage
  Dockerfile (non-root + healthcheck), README API/Docker docs. 47 tests green,
  lint clean. Next: M5 (C4 architecture docs + polish).
- 2026-06-12 — M5 complete: full C4 architecture docs (context/container/
  component/code, Mermaid) + 3 ADRs, CONTRIBUTING, CHANGELOG, README badges,
  Voyage embedder, factory tests (which caught & fixed a Settings alias bug).
  51 tests green, lint clean. **Planned roadmap M1–M5 done.** Next: M6 backlog
  (hardening: reranking, streaming, persisted API index, examples).
- 2026-06-12 — M6 (part 1): two-stage retrieval — `Reranker` protocol + hybrid
  BM25+embedding `LexicalReranker`, wired into Retriever/Pipeline (opt-in via
  config), ADR 0004. Tightened mypy to fully strict (fixed 6 issues) and made it
  blocking in CI. 59 tests green, ruff + mypy clean. Next: streaming `/ask`,
  examples, persisted API index.
- 2026-06-12 — M6 (part 2): persisted API index (RAGFORGE_INDEX_PATH load on
  startup + auto-save after ingest + `/save` endpoint); runnable
  `examples/quickstart.py` end-to-end demo (ASCII-safe output). 63 tests green,
  ruff + mypy clean. Next: streaming `/ask`, coverage gate.
- 2026-06-12 — M6 (part 3): `LLMReranker` (Claude-as-judge, single structured
  JSON call, safe fallback) behind the same `Reranker` protocol; selectable via
  `rerank_provider=lexical|llm`. ADR 0004 updated. 72 tests green, ruff + mypy
  clean. Next: streaming `/ask`, coverage gate.
- 2026-06-12 — M6 (part 4): streaming. Refactored the agent loop into
  `iter_events()` (single source of truth; `answer()` drains it) emitting
  search/results/answer `AgentEvent`s; added SSE `POST /ask/stream`. 76 tests
  green, ruff + mypy clean. Next: coverage gate, CLI streaming.
- 2026-06-12 — M6 (part 5): coverage gate. Added CLI test suite (caught & fixed
  two missing-file error paths in ingest/query), configured coverage to omit the
  network-only adapters, set `fail_under=90`. Coverage 95%, 84 tests green,
  ruff + mypy clean. Next: CLI streaming output, cross-encoder reranker.
- 2026-06-12 — M6 (part 6): `ask --stream` consumes the agent's iter_events and
  prints live search/retrieved progress before the answer (ASCII-safe output).
  85 tests green, ruff + mypy clean. Next: cross-encoder reranker, hybrid
  dense+sparse first-pass retrieval.
- 2026-06-12 — M6 (part 7): hybrid retrieval. BM25 `sparse` index +
  Reciprocal Rank Fusion in the Retriever (dense -> hybrid -> rerank pipeline),
  config-gated, rebuilt from persisted chunks on load. ADR 0005. Coverage 95%,
  92 tests green, ruff + mypy clean. Next: cross-encoder reranker, persist sparse.
- 2026-06-12 — Docs: refreshed the C4 model (component + code diagrams +
  overview) to include the rerank package, sparse/hybrid retrieval, and the
  agent event stream — keeping the flagship architecture doc accurate as the
  system grew. Docs-only; 92 tests green.
