# ragforge

[![CI](https://github.com/cnfuse/ragforge/actions/workflows/ci.yml/badge.svg)](https://github.com/cnfuse/ragforge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Built on Claude](https://img.shields.io/badge/LLM-Claude%20Opus%204.8-8A2BE2)](https://www.anthropic.com)

> An agentic Retrieval-Augmented Generation platform with a built-in evaluation harness, built on Claude.

`ragforge` is a compact but production-shaped RAG system. It ingests documents,
retrieves relevant context, and (when an API key is present) answers questions
with **Claude** while citing its sources — and it ships with an **evaluation
harness** so retrieval and answer quality are measured, not assumed.

It is designed to run **fully offline by default**: the standard embedder is a
local, deterministic feature-hasher, so you can index a corpus and retrieve
against it with no network access and no API key. Plug in a real embedding
provider and the Anthropic API when you want production-grade semantics and
generated answers.

---

## Why this project

This repository is a portfolio piece demonstrating the breadth of the **AI
Engineer** role end to end:

- **RAG pipeline** — ingestion, paragraph-aware chunking, embeddings, a vector
  store, ranked retrieval, optional **hybrid dense+sparse retrieval** (BM25 fused
  with embeddings via Reciprocal Rank Fusion), optional **reranking** (hybrid
  BM25+embedding or Claude-as-judge), and optional **MMR diversification** —
  behind clean, swappable interfaces.
- **LLM integration** — Claude via the official Anthropic SDK, with adaptive
  thinking and graceful offline fallback.
- **Agentic tool use** — the model decides when to search the corpus rather than
  answering from memory.
- **Evaluation** — retrieval metrics (hit-rate, MRR, recall@k) and
  answer-grounding checks, so changes are judged against numbers.
- **Engineering rigour** — typed config, structured logging, a test suite, CI,
  a CLI, an HTTP API, and **C4 architecture documentation**.

## Architecture at a glance

```
            ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌───────────┐
documents → │ Ingestion │ → │ Embedder │ → │ VectorStore │ ← │ Retriever │
            └──────────┘   └──────────┘   └────────────┘   └─────┬─────┘
                                                                  │ top-k chunks
                                                            ┌─────▼─────┐
                                            question  ────► │ RAG Agent │ ──► answer + citations
                                                            │ (Claude)  │
                                                            └───────────┘
```

Full C4 model (Context → Container → Component → Code), with Mermaid diagrams,
lives in [`docs/architecture/`](docs/architecture/); design decisions are
recorded as ADRs in [`docs/adr/`](docs/adr/).

## Try it in 10 seconds

```bash
pip install -e ".[dev,api]"
python examples/quickstart.py     # ingest → query → ask → evaluate, fully offline
```

## Install

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -e ".[dev,api]"
```

## Quick start (offline, no API key)

```bash
# Index a directory of text/markdown files
ragforge ingest ./docs --index my_index.json

# Ask the index for the most relevant passages
ragforge query "how does chunking work?" --index my_index.json --top-k 5

# Let the agent answer (Claude if ANTHROPIC_API_KEY is set, else offline mock)
ragforge ask "how does chunking work?" --index my_index.json

# Stream the agent's progress (searching -> retrieved -> answer)
ragforge ask "how does chunking work?" --index my_index.json --stream
```

## Evaluation

Retrieval and answer quality are measured, not assumed:

```bash
ragforge eval --corpus data/sample/corpus --dataset data/sample/qa.jsonl --top-k 3
# retrieval: hit_rate@3=1.000  recall@3=1.000  mrr=1.000  ndcg@3=1.000
# answers:   expected_match=0.750  grounding=0.887
```

Metrics: hit-rate, recall@k, MRR, nDCG for retrieval; expected-substring match
and citation grounding for answers. Add `--json-out report.json` for a full
per-query report.

## HTTP API

```bash
pip install -e ".[api]"
uvicorn ragforge.api.app:app --reload          # http://127.0.0.1:8000/docs

# or with Docker
docker build -t ragforge . && docker run -p 8000:8000 ragforge
```

| Method | Path      | Purpose                                   |
| ------ | --------- | ----------------------------------------- |
| GET    | `/health` | Liveness, index size, model info          |
| POST   | `/ingest` | Add documents to the index                |
| POST   | `/query`  | Retrieve relevant chunks                  |
| POST   | `/ask`    | Answer a question with the agent          |
| POST   | `/ask/stream` | Stream the agent's progress as SSE    |
| POST   | `/save`   | Persist the current index to disk         |

Interactive OpenAPI docs are served at `/docs`. Set `RAGFORGE_INDEX_PATH` to make
the index durable: the service loads it on startup and auto-saves after each
ingest, so it survives restarts.

## Configuration

All settings are environment-driven (prefix `RAGFORGE_`). Copy
[`.env.example`](.env.example) to `.env` and adjust. Set `ANTHROPIC_API_KEY` to
unlock Claude-backed answers.

## Development

```bash
pytest            # run the test suite (coverage gate: fail under 90%)
ruff check .      # lint
mypy              # type-check (strict)
```

The suite is hermetic — no network or API key — and currently sits at ~95%
coverage. The two thin external adapters (Anthropic, Voyage) are exercised by
integration, not unit tests, and are excluded from the gate.

## Status & roadmap

This project is built incrementally; see [`ROADMAP.md`](ROADMAP.md) for the
current state and what's next.

## License

MIT — see [`LICENSE`](LICENSE).
