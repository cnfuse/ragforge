# Contributing to ragforge

Thanks for your interest. This guide covers the local workflow and the
conventions the project follows.

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -e ".[dev,api]"
```

## Quality gates

All of these must pass before a change is merged; CI runs them on Python
3.10–3.12.

```bash
pytest --cov=ragforge --cov-report=term-missing   # tests + coverage
ruff check .                                       # lint
mypy                                               # type check
```

### Conventions

- **Offline by default.** New external dependencies must sit behind a protocol
  with a deterministic, offline default so the test suite stays hermetic (no
  network, no API key). See [ADR 0002](docs/adr/0002-offline-first-defaults.md).
- **Protocols over base classes** for swappable components (`Embedder`,
  `VectorStore`, `LLM`).
- **Tests for every component.** Prefer the `MockLLM` and `HashingEmbedder` in
  tests; never call live providers from the suite.
- **Type hints everywhere**; keep `mypy` clean.
- **Docstrings** explain *why*, not just *what*; match the surrounding style.

## Commits & PRs

- Use clear, imperative commit subjects (`feat:`, `fix:`, `docs:`, `test:` …).
- Keep changes small and coherent; update `ROADMAP.md` and `CHANGELOG.md` when
  relevant.
- Record significant design decisions as an ADR in `docs/adr/`.

## Architecture

Read [`docs/architecture/`](docs/architecture/) (the C4 model) before making
structural changes.
