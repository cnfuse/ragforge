"""Claude integration via the official Anthropic SDK.

Uses adaptive thinking and the ``effort`` control (both GA on Opus 4.8) and
exposes the same :class:`LLM` interface as the mock. The Anthropic client
resolves its credentials from the environment (``ANTHROPIC_API_KEY``); we never
handle the key directly.
"""

from __future__ import annotations

from typing import Any

from ragforge.llm.base import LLMResponse, ToolCall, ToolSpec
from ragforge.logging import get_logger

log = get_logger("llm.anthropic")


class AnthropicLLM:
    """Adapter from the Anthropic Messages API to the :class:`LLM` protocol."""

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        max_tokens: int = 2048,
        effort: str = "high",
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "The 'anthropic' package is required for live LLM calls; "
                "install it with `pip install anthropic`."
            ) from exc
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(max_tokens, self.max_tokens),
            "system": system,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
        }
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]

        response = self._client.messages.create(**kwargs)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))

        return LLMResponse(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            model=response.model,
        )
