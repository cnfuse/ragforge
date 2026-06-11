"""A retrieval-augmented agent that lets the model decide when to search.

Rather than always stuffing top-k chunks into the prompt, the agent exposes a
``search_corpus`` tool and runs a bounded tool-use loop: the model searches when
it judges it needs evidence, reads the results, and answers with citations. Every
chunk surfaced during the run is collected so the final :class:`Answer` carries
the evidence it was grounded in.

The loop is provider-agnostic — it works identically against the live Claude
client and the offline :class:`~ragforge.llm.mock.MockLLM`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ragforge.llm.base import LLM, ToolCall, ToolSpec
from ragforge.logging import get_logger
from ragforge.pipeline import Pipeline
from ragforge.types import AgentEvent, Answer, ScoredChunk

log = get_logger("agent")

SYSTEM_PROMPT = (
    "You are ragforge, a precise question-answering assistant. Answer the user's "
    "question using ONLY information returned by the search_corpus tool. Search "
    "before answering. If the corpus does not contain the answer, say so plainly "
    "rather than guessing. Cite the supporting passages by their [n] markers."
)

_SEARCH_TOOL = ToolSpec(
    name="search_corpus",
    description=(
        "Search the indexed document corpus for passages relevant to a query. "
        "Returns ranked snippets, each tagged with an [n] marker and its source "
        "document id. Call this whenever you need evidence to answer."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {
                "type": "integer",
                "description": "How many passages to return (default 5).",
            },
        },
        "required": ["query"],
    },
)


class RagAgent:
    """Drives a bounded search-and-answer loop over a :class:`Pipeline`."""

    def __init__(self, pipeline: Pipeline, llm: LLM, *, max_steps: int = 4) -> None:
        self.pipeline = pipeline
        self.llm = llm
        self.max_steps = max_steps

    def answer(self, question: str) -> Answer:
        """Answer ``question``, searching the corpus as needed, with citations.

        Thin wrapper over :meth:`iter_events`: drains the event stream and
        returns the :class:`Answer` carried by the terminal event.
        """
        final: Answer | None = None
        for event in self.iter_events(question):
            if event.answer is not None:
                final = event.answer
        assert final is not None  # iter_events always emits a terminal answer
        return final

    def iter_events(self, question: str) -> Iterator[AgentEvent]:
        """Run the agent loop, yielding progress events as they happen.

        Yields ``search`` and ``results`` events per tool call and a terminal
        ``answer`` (or ``budget_exhausted``) event carrying the final
        :class:`Answer`. This is the single source of truth for the loop;
        :meth:`answer` is built on top of it.
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        collected: dict[str, ScoredChunk] = {}

        for step in range(self.max_steps):
            response = self.llm.generate(
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=[_SEARCH_TOOL],
                max_tokens=self.pipeline.settings.max_tokens,
            )

            if not response.wants_tools:
                yield AgentEvent(
                    type="answer",
                    step=step + 1,
                    answer=Answer(
                        question=question,
                        text=response.text or "(no answer produced)",
                        citations=self._rank(collected),
                        model=response.model,
                    ),
                )
                return

            # Reconstruct the assistant turn so the conversation stays valid.
            assistant_content: list[dict[str, Any]] = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for call in response.tool_calls:
                assistant_content.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
                )
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute each requested tool call, emitting events, and feed results back.
            results_content: list[dict[str, Any]] = []
            for call in response.tool_calls:
                query = str(call.input.get("query", "")).strip()
                yield AgentEvent(type="search", step=step + 1, message=query)
                rendered, n_hits = self._run_tool(call, collected)
                yield AgentEvent(
                    type="results", step=step + 1, message=query, data={"count": n_hits}
                )
                results_content.append(
                    {"type": "tool_result", "tool_use_id": call.id, "content": rendered}
                )
            messages.append({"role": "user", "content": results_content})
            log.info("step %d: ran %d tool call(s)", step + 1, len(response.tool_calls))

        # Loop budget exhausted without a final answer.
        yield AgentEvent(
            type="budget_exhausted",
            step=self.max_steps,
            answer=Answer(
                question=question,
                text="I was unable to converge on an answer within the step budget.",
                citations=self._rank(collected),
                model=getattr(self.llm, "model", None),
            ),
        )

    def _run_tool(
        self, call: ToolCall, collected: dict[str, ScoredChunk]
    ) -> tuple[str, int]:
        """Execute a tool call; return (rendered text for the LLM, hit count)."""
        if call.name != _SEARCH_TOOL.name:
            return f"Error: unknown tool {call.name!r}.", 0
        query = str(call.input.get("query", "")).strip()
        top_k = int(call.input.get("top_k") or self.pipeline.settings.top_k)
        if not query:
            return "Error: 'query' is required.", 0
        hits = self.pipeline.retrieve(query, top_k=top_k)
        if not hits:
            return "No relevant passages found in the corpus.", 0
        for hit in hits:
            # Keep the best score seen for each chunk across all searches.
            prev = collected.get(hit.chunk.id)
            if prev is None or hit.score > prev.score:
                collected[hit.chunk.id] = hit
        return self._render(hits), len(hits)

    @staticmethod
    def _render(hits: list[ScoredChunk]) -> str:
        lines = []
        for i, hit in enumerate(hits, start=1):
            text = hit.chunk.text.replace("\n", " ").strip()
            lines.append(f"[{i}] (doc={hit.chunk.doc_id}, score={hit.score:.3f}) {text}")
        return "\n".join(lines)

    @staticmethod
    def _rank(collected: dict[str, ScoredChunk]) -> list[ScoredChunk]:
        return sorted(collected.values(), key=lambda s: s.score, reverse=True)
