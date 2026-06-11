"""Tests for the deterministic offline MockLLM."""

from __future__ import annotations

from ragforge.llm.base import ToolSpec
from ragforge.llm.mock import MockLLM

SEARCH = ToolSpec(
    name="search_corpus",
    description="search",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
)


def test_requests_search_when_tool_available() -> None:
    llm = MockLLM()
    resp = llm.generate(
        system="s",
        messages=[{"role": "user", "content": "what is rust?"}],
        tools=[SEARCH],
    )
    assert resp.wants_tools
    assert resp.tool_calls[0].name == "search_corpus"
    assert resp.tool_calls[0].input["query"] == "what is rust?"


def test_answers_from_tool_results() -> None:
    llm = MockLLM()
    messages = [
        {"role": "user", "content": "what is rust?"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "x", "name": "search_corpus", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "x",
                    "content": "[1] Rust is a systems language.",
                }
            ],
        },
    ]
    resp = llm.generate(system="s", messages=messages, tools=[SEARCH])
    assert not resp.wants_tools
    assert resp.stop_reason == "end_turn"
    assert "Rust is a systems language" in resp.text


def test_offline_answer_without_tools() -> None:
    llm = MockLLM()
    resp = llm.generate(system="s", messages=[{"role": "user", "content": "hi"}], tools=None)
    assert resp.stop_reason == "end_turn"
    assert not resp.tool_calls
