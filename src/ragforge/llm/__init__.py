"""Language-model providers and the shared LLM contract.

The :class:`LLM` protocol is provider-agnostic. :class:`AnthropicLLM` calls
Claude through the official SDK; :class:`MockLLM` is a deterministic, offline
implementation that drives the same tool-use loop so the agent can be tested
without a network or API key.
"""

from __future__ import annotations

from ragforge.config import Settings
from ragforge.llm.base import LLM, LLMResponse, ToolCall, ToolSpec
from ragforge.llm.mock import MockLLM

__all__ = ["LLM", "LLMResponse", "ToolCall", "ToolSpec", "MockLLM", "build_llm"]


def build_llm(settings: Settings) -> LLM:
    """Return a live Claude client if an API key is configured, else the mock."""
    if settings.has_live_llm:
        from ragforge.llm.anthropic_client import AnthropicLLM

        return AnthropicLLM(
            model=settings.model,
            max_tokens=settings.max_tokens,
            effort=settings.effort,
        )
    return MockLLM()
