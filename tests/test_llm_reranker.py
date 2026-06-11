"""Tests for the LLM-as-judge reranker using deterministic fake LLMs."""

from __future__ import annotations

from typing import Any

import pytest

from ragforge.config import Settings
from ragforge.llm.base import LLMResponse, ToolSpec
from ragforge.rerank import build_reranker
from ragforge.rerank.llm_reranker import LLMReranker
from ragforge.types import Chunk, ScoredChunk


def _cand(doc_id: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=doc_id, doc_id=doc_id, text=text), score=score)


class _ScoringLLM:
    """Returns fixed relevance scores as JSON; records the prompt it saw."""

    model = "fake"

    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = scores
        self.last_prompt: str | None = None

    def generate(self, *, system: str, messages: list[dict[str, Any]], tools=None, max_tokens=2048):  # type: ignore[no-untyped-def]
        self.last_prompt = messages[-1]["content"]
        payload = {"scores": [{"index": i, "relevance": r} for i, r in self.scores.items()]}
        import json

        return LLMResponse(text=json.dumps(payload), stop_reason="end_turn", model=self.model)


class _GarbageLLM:
    model = "garbage"

    def generate(self, *, system, messages, tools=None, max_tokens=2048):  # type: ignore[no-untyped-def]
        return LLMResponse(text="sorry, I cannot do that", stop_reason="end_turn")


def test_empty_candidates() -> None:
    assert LLMReranker(_ScoringLLM({})).rerank("q", [], top_k=3) == []


def test_reorders_by_llm_relevance() -> None:
    cands = [_cand("a", "alpha", 0.9), _cand("b", "beta", 0.8), _cand("c", "gamma", 0.7)]
    # First-pass order is a > b > c; the judge flips it to c > b > a.
    llm = _ScoringLLM({0: 0.1, 1: 0.5, 2: 0.95})
    out = LLMReranker(llm).rerank("q", cands, top_k=3)
    assert [s.chunk.doc_id for s in out] == ["c", "b", "a"]
    assert out[0].score == pytest.approx(0.95)


def test_prompt_lists_every_passage() -> None:
    cands = [_cand("a", "alpha", 0.9), _cand("b", "beta", 0.8)]
    llm = _ScoringLLM({0: 0.2, 1: 0.3})
    LLMReranker(llm).rerank("find beta", cands, top_k=2)
    assert llm.last_prompt is not None
    assert "[0]" in llm.last_prompt and "[1]" in llm.last_prompt
    assert "find beta" in llm.last_prompt


def test_truncates_to_top_k() -> None:
    cands = [_cand(str(i), f"t{i}", 0.5) for i in range(5)]
    llm = _ScoringLLM({i: float(i) / 5 for i in range(5)})
    out = LLMReranker(llm).rerank("q", cands, top_k=2)
    assert len(out) == 2


def test_unparseable_output_falls_back_to_original_order() -> None:
    cands = [_cand("a", "alpha", 0.9), _cand("b", "beta", 0.1)]
    out = LLMReranker(_GarbageLLM()).rerank("q", cands, top_k=2)
    assert [s.chunk.doc_id for s in out] == ["a", "b"]  # unchanged


def test_partial_scores_keep_first_pass_for_missing() -> None:
    cands = [_cand("a", "alpha", 0.9), _cand("b", "beta", 0.2)]
    # Only score index 1; index 0 should retain its first-pass score (0.9).
    out = LLMReranker(_ScoringLLM({1: 0.95})).rerank("q", cands, top_k=2)
    assert [s.chunk.doc_id for s in out] == ["b", "a"]


def test_build_reranker_selects_llm_provider() -> None:
    reranker = build_reranker(Settings(rerank_enabled=True, rerank_provider="llm"))
    assert isinstance(reranker, LLMReranker)


def test_build_reranker_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        build_reranker(Settings(rerank_enabled=True, rerank_provider="nope"))


def test_protocol_compat() -> None:
    spec = ToolSpec("x", "y", {"type": "object"})  # smoke: import surface intact
    assert spec.name == "x"
