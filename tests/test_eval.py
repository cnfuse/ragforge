"""Tests for the evaluation harness: metrics, dataset loading, and the runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragforge.agent.rag_agent import RagAgent
from ragforge.eval.answer_metrics import expected_match, grounding
from ragforge.eval.dataset import QAExample, load_dataset
from ragforge.eval.retrieval_metrics import (
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragforge.eval.runner import evaluate_answers, evaluate_retrieval
from ragforge.llm.mock import MockLLM
from ragforge.pipeline import Pipeline
from ragforge.types import Answer, Chunk, Document, ScoredChunk

# --- retrieval metrics ----------------------------------------------------


def test_hit_at_k() -> None:
    assert hit_at_k(["a", "b", "c"], {"c"}, k=3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0
    assert hit_at_k(["a"], set(), k=3) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, k=3) == 0.5


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["x", "y", "rel"], {"rel"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["rel"], {"rel"}) == 1.0
    assert reciprocal_rank(["x"], {"rel"}) == 0.0


def test_ndcg_rewards_higher_ranking() -> None:
    top = ndcg_at_k(["rel", "x", "y"], {"rel"}, k=3)
    low = ndcg_at_k(["x", "y", "rel"], {"rel"}, k=3)
    assert top == 1.0
    assert top > low > 0.0


# --- dataset --------------------------------------------------------------


def test_load_dataset_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text(
        '{"id": "a", "question": "q?", "relevant_doc_ids": ["d1"]}\n\n'
        '{"id": "b", "question": "q2?", "expected_substrings": ["x"]}\n',
        encoding="utf-8",
    )
    examples = load_dataset(path)
    assert [e.id for e in examples] == ["a", "b"]
    assert examples[0].relevant_doc_ids == ["d1"]


def test_load_dataset_rejects_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


# --- answer metrics -------------------------------------------------------


def _answer(text: str, cited: str | None) -> Answer:
    citations = []
    if cited is not None:
        citations = [ScoredChunk(chunk=Chunk(id="c", doc_id="d", text=cited), score=1.0)]
    return Answer(question="q", text=text, citations=citations)


def test_expected_match() -> None:
    ans = _answer("Rust ensures memory safety.", cited=None)
    assert expected_match(ans, ["memory safety"]) == 1.0
    assert expected_match(ans, ["garbage collector"]) == 0.0
    assert expected_match(ans, []) == 1.0


def test_grounding_rewards_supported_claims() -> None:
    grounded = _answer("Rust memory safety", cited="Rust guarantees memory safety")
    partial = _answer("Rust ensures memory safety", cited="Rust guarantees memory safety")
    ungrounded = _answer("Rust ensures memory safety", cited=None)
    assert grounding(grounded) == 1.0  # every content word supported
    assert 0.0 < grounding(partial) < 1.0  # "ensures" not in the citation
    assert grounding(ungrounded) == 0.0


# --- runner ---------------------------------------------------------------

CORPUS = [
    Document(id="py", text="Python is a readable high-level programming language."),
    Document(id="rust", text="Rust is a systems language with memory safety guarantees."),
]
DATASET = [
    QAExample(id="1", question="memory safe systems language", relevant_doc_ids=["rust"]),
    QAExample(id="2", question="readable high level language", relevant_doc_ids=["py"]),
]


def test_evaluate_retrieval_perfect_on_easy_set() -> None:
    pipeline = Pipeline()
    pipeline.ingest(CORPUS)
    report = evaluate_retrieval(pipeline, DATASET, k=2)
    assert report.num_examples == 2
    assert report.retrieval.hit_rate == 1.0
    assert report.retrieval.mrr == 1.0


def test_evaluate_answers_includes_grounding() -> None:
    pipeline = Pipeline()
    pipeline.ingest(CORPUS)
    agent = RagAgent(pipeline, MockLLM())
    report = evaluate_answers(pipeline, agent, DATASET, k=2)
    assert report.mean_expected_match is not None
    assert report.mean_grounding is not None
    assert len(report.answers) == 2
