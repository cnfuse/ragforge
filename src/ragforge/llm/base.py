"""Provider-agnostic LLM contract.

Messages use the Anthropic content-block shape (a list of ``{"role", "content"}``
dicts) so the live client passes them through unchanged and the mock can parse
them deterministically. A response carries either final text, one or more tool
calls, or both — the agent loop reacts to ``stop_reason``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call, described by a JSON Schema for its input."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """The model's reply for one turn of the conversation."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    model: str | None = None

    @property
    def wants_tools(self) -> bool:
        """True when the model is asking to run tools before continuing."""
        return self.stop_reason == "tool_use" and bool(self.tool_calls)


class LLM(Protocol):
    """Generate a single assistant turn, optionally with tools available."""

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse: ...
