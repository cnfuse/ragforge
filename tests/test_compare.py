"""Tests for the retrieval-configuration comparison (ablation) harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragforge.config import Settings
from ragforge.eval.compare import (
    compare_retrieval_configs,
    default_matrix,
    format_comparison,
)
from ragforge.eval.dataset import QAExample
from ragforge.types import Document

DOCS = [
    Document(id="py", text="Python is a readable high-level programming language."),
    Document(id="rust", text="Rust uses the cargo build tool and guarantees memory safety."),
    Document(id="go", text="Go is a compiled language with built-in goroutines."),
]
DATASET = [
    QAExample(id="1", question="cargo build tool", relevant_doc_ids=["rust"]),
    QAExample(id="2", question="readable high level language", relevant_doc_ids=["py"]),
]


def test_default_matrix_has_expected_configs() -> None:
    matrix = default_matrix()
    assert set(matrix) == {"dense", "hybrid", "hybrid+rerank", "hybrid+mmr"}
    assert matrix["hybrid"].hybrid_enabled is True


def test_compare_returns_metrics_per_config() -> None:
    configs = {"dense": Settings(), "hybrid": Settings(hybrid_enabled=True)}
    results = compare_retrieval_configs(DOCS, DATASET, configs, k=3)
    assert set(results) == {"dense", "hybrid"}
    for res in results.values():
        assert res.metrics.k == 3
        assert 0.0 <= res.metrics.ndcg <= 1.0
        assert res.mean_latency_ms >= 0.0


def test_format_comparison_renders_table_and_marks_best() -> None:
    results = compare_retrieval_configs(DOCS, DATASET, default_matrix(), k=3)
    table = format_comparison(results)
    assert all(col in table for col in ("config", "hit_rate", "ndcg", "ms/query"))
    for name in default_matrix():
        assert name in table
    assert "*" in table  # best config is marked


def test_format_comparison_empty() -> None:
    assert "no configurations" in format_comparison({})


def test_cli_eval_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from ragforge.cli import main

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "rust.txt").write_text(
        "Rust uses the cargo build tool and guarantees memory safety.", encoding="utf-8"
    )
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        '{"id": "1", "question": "cargo build tool", "relevant_doc_ids": ["rust.txt"]}\n',
        encoding="utf-8",
    )
    code = main(
        ["eval", "--corpus", str(corpus), "--dataset", str(dataset), "--top-k", "3", "--compare"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "hybrid+rerank" in out
