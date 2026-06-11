# ragforge

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
  store, and ranked retrieval, behind clean, swappable interfaces.
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

Full C4 model (Context → Container → Component → Code) lives in
[`docs/architecture/`](docs/architecture/).

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
```

## Configuration

All settings are environment-driven (prefix `RAGFORGE_`). Copy
[`.env.example`](.env.example) to `.env` and adjust. Set `ANTHROPIC_API_KEY` to
unlock Claude-backed answers.

## Development

```bash
pytest            # run the test suite
ruff check .      # lint
mypy              # type-check
```

## Status & roadmap

This project is built incrementally; see [`ROADMAP.md`](ROADMAP.md) for the
current state and what's next.

## License

MIT — see [`LICENSE`](LICENSE).
