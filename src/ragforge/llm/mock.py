"""A deterministic, offline LLM used as the default and in tests.

It implements the same tool-use protocol as the live client so the agent loop is
exercised identically with or without a network:

1. Given a question and a ``search_corpus`` tool, it requests the tool call.
2. Given the tool results, it returns a grounded answer that quotes the
   highest-ranked retrieved passage.

The behaviour is fully deterministic, which makes agent tests reproducible.
"""

from __future__ import annotations

from typing import Any

from ragforge.llm.base import LLMResponse, ToolCall, ToolSpec

_SEARCH_TOOL = "search_corpus"


def _last_user_text(content: Any) -> str:
    """Extract plain text from a message's content (string or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return " ".join(p for p in parts if p)
    return ""


def _tool_results(content: Any) -> list[str]:
    """Collect tool_result payloads from a message's content blocks."""
    if not isinstance(content, list):
        return []
    out: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            payload = block.get("content", "")
            if isinstance(payload, list):
                payload = " ".join(
                    b.get("text", "") for b in payload if isinstance(b, dict)
                )
            out.append(str(payload))
    return out


class MockLLM:
    """Offline stand-in that follows the tool-use loop deterministically."""

    model = "mock-llm"

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        last = messages[-1]
        content = last.get("content")

        # Phase 2: we have tool results -> produce a grounded final answer.
        results = _tool_results(content)
        if results:
            snippet = results[0].strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:280].rsplit(" ", 1)[0] + "…"
            if snippet:
                answer = f"Based on the retrieved context: {snippet}"
            else:
                answer = "I couldn't find anything relevant in the corpus."
            return LLMResponse(text=answer, stop_reason="end_turn", model=self.model)

        # Phase 1: a question with a search tool available -> ask to search.
        question = _last_user_text(content)
        has_search = bool(tools) and any(t.name == _SEARCH_TOOL for t in tools)
        if has_search and question:
            call = ToolCall(id="mock-call-1", name=_SEARCH_TOOL, input={"query": question})
            return LLMResponse(
                text="",
                tool_calls=[call],
                stop_reason="tool_use",
                model=self.model,
            )

        # No tools available: answer offline without retrieval.
        return LLMResponse(
            text=(
                "Running offline without retrieval; set ANTHROPIC_API_KEY for "
                f"Claude-backed answers. Question was: {question}"
            ),
            stop_reason="end_turn",
            model=self.model,
        )
