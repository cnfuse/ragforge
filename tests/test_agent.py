"""Tests for the agentic RAG loop using the offline MockLLM."""

from __future__ import annotations

from ragforge.agent.rag_agent import RagAgent
from ragforge.llm.base import LLM, LLMResponse, ToolSpec
from ragforge.llm.mock import MockLLM
from ragforge.pipeline import Pipeline
from ragforge.types import Document

CORPUS = [
    Document(id="py", text="Python is a high-level programming language with readable syntax."),
    Document(id="rust", text="Rust is a systems programming language focused on memory safety."),
    Document(id="cook", text="Risotto needs arborio rice, warm stock, and constant stirring."),
]


def _pipeline() -> Pipeline:
    p = Pipeline()
    p.ingest(CORPUS)
    return p


def test_agent_searches_then_answers_with_citations() -> None:
    agent = RagAgent(_pipeline(), MockLLM())
    answer = agent.answer("which language is focused on memory safety?")
    assert "memory safety" in answer.text.lower()
    assert answer.citations
    assert answer.citations[0].chunk.doc_id == "rust"
    assert answer.model == "mock-llm"


def test_agent_returns_evidence_sorted_by_score() -> None:
    agent = RagAgent(_pipeline(), MockLLM())
    answer = agent.answer("readable programming language")
    scores = [c.score for c in answer.citations]
    assert scores == sorted(scores, reverse=True)


class _NoToolLLM:
    """An LLM that never calls tools — exercises the direct-answer path."""

    model = "no-tool"

    def generate(self, *, system, messages, tools=None, max_tokens=2048):  # type: ignore[no-untyped-def]
        return LLMResponse(text="Direct answer.", stop_reason="end_turn", model=self.model)


def test_agent_handles_model_that_skips_search() -> None:
    agent = RagAgent(_pipeline(), _NoToolLLM())
    answer = agent.answer("anything")
    assert answer.text == "Direct answer."
    assert answer.citations == []


class _AlwaysSearchLLM:
    """An LLM that always asks to search — exercises the step-budget guard."""

    model = "loop"

    def generate(self, *, system, messages, tools=None, max_tokens=2048):  # type: ignore[no-untyped-def]
        from ragforge.llm.base import ToolCall

        return LLMResponse(
            text="",
            tool_calls=[ToolCall(id="c", name="search_corpus", input={"query": "x"})],
            stop_reason="tool_use",
            model=self.model,
        )


def test_agent_respects_step_budget() -> None:
    agent = RagAgent(_pipeline(), _AlwaysSearchLLM(), max_steps=2)
    answer = agent.answer("loop forever?")
    assert "step budget" in answer.text


def test_mock_is_an_llm() -> None:
    # Structural check that MockLLM satisfies the protocol surface.
    llm: LLM = MockLLM()
    assert isinstance(ToolSpec("a", "b", {"type": "object"}).name, str)
    assert llm.generate(system="", messages=[{"role": "user", "content": "hi"}]).stop_reason
