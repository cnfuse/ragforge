"""Tests for the command-line interface (ingest / query / ask / eval)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragforge.cli import main

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_CORPUS = _ROOT / "data" / "sample" / "corpus"
_SAMPLE_DATASET = _ROOT / "data" / "sample" / "qa.jsonl"


def _write_corpus(tmp_path: Path) -> Path:
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "Rust is a systems programming language with memory safety guarantees. "
        "It uses the cargo build tool.",
        encoding="utf-8",
    )
    return doc


def test_ingest_then_query(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    doc = _write_corpus(tmp_path)
    index = tmp_path / "idx.json"

    assert main(["ingest", str(doc), "--index", str(index)]) == 0
    assert index.exists()

    assert main(["query", "memory safety", "--index", str(index), "--top-k", "2"]) == 0
    out = capsys.readouterr().out
    assert "score=" in out
    assert "doc.txt" in out


def test_ask_uses_offline_agent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    doc = _write_corpus(tmp_path)
    index = tmp_path / "idx.json"
    main(["ingest", str(doc), "--index", str(index)])

    assert main(["ask", "what build tool does rust use?", "--index", str(index)]) == 0
    out = capsys.readouterr().out
    assert out.strip()  # an answer was printed


def test_ask_stream_prints_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = _write_corpus(tmp_path)
    index = tmp_path / "idx.json"
    main(["ingest", str(doc), "--index", str(index)])

    code = main(["ask", "what build tool does rust use?", "--index", str(index), "--stream"])
    assert code == 0
    out = capsys.readouterr().out
    assert "searching:" in out
    assert "retrieved" in out
    assert out.strip()


def test_query_missing_index_returns_error(tmp_path: Path) -> None:
    assert main(["query", "x", "--index", str(tmp_path / "nope.json")]) == 1


def test_ingest_nonexistent_path_returns_error(tmp_path: Path) -> None:
    assert main(["ingest", str(tmp_path / "missing")]) == 1


def test_ingest_directory_without_text_files(tmp_path: Path) -> None:
    (tmp_path / "image.bin").write_bytes(b"\x00\x01")
    assert main(["ingest", str(tmp_path), "--index", str(tmp_path / "i.json")]) == 1


def test_eval_retrieval_only_with_json_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "report.json"
    code = main(
        [
            "eval",
            "--corpus",
            str(_SAMPLE_CORPUS),
            "--dataset",
            str(_SAMPLE_DATASET),
            "--top-k",
            "3",
            "--retrieval-only",
            "--json-out",
            str(report_path),
        ]
    )
    assert code == 0
    assert report_path.exists()
    out = capsys.readouterr().out
    assert "retrieval:" in out


def test_eval_full_answers(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["eval", "--corpus", str(_SAMPLE_CORPUS), "--dataset", str(_SAMPLE_DATASET)]
    )
    assert code == 0
    assert "answers:" in capsys.readouterr().out


def test_no_command_exits_nonzero() -> None:
    with pytest.raises(SystemExit):
        main([])
