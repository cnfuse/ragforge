"""An LLM-as-judge reranker.

Asks the language model to score each candidate's relevance to the query in a
single call, returning structured JSON. This is the highest-precision rerank
option — the model reads each passage and judges topical relevance directly —
at the cost of one extra LLM call per query.

It degrades safely: if the model's output can't be parsed, the original
first-pass order is preserved rather than raising. It shares the :class:`LLM`
protocol, so the offline mock (or any deterministic test double) drives it
without a network.
"""

from __future__ import annotations

import json
import re

from ragforge.llm.base import LLM
from ragforge.logging import get_logger
from ragforge.types import ScoredChunk

log = get_logger("rerank.llm")

_SYSTEM = (
    "You are a search relevance judge. Given a query and numbered passages, rate "
    "how well each passage answers the query from 0.0 (irrelevant) to 1.0 "
    "(directly answers). Respond ONLY with JSON of the form "
    '{"scores": [{"index": <int>, "relevance": <float>}, ...]} covering every '
    "passage index, and nothing else."
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class LLMReranker:
    """Reranks candidates by LLM-judged relevance, with a safe fallback."""

    def __init__(self, llm: LLM, *, max_tokens: int = 1024) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        prompt = self._build_prompt(query, candidates)
        response = self.llm.generate(
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            max_tokens=self.max_tokens,
        )
        scores = self._parse_scores(response.text, n=len(candidates))
        if scores is None:
            log.warning("LLM reranker could not parse scores; keeping first-pass order")
            return candidates[:top_k]

        rescored = [
            ScoredChunk(chunk=cand.chunk, score=scores.get(i, cand.score))
            for i, cand in enumerate(candidates)
        ]
        rescored.sort(key=lambda s: s.score, reverse=True)
        return rescored[:top_k]

    @staticmethod
    def _build_prompt(query: str, candidates: list[ScoredChunk]) -> str:
        lines = [f"Query: {query}", "", "Passages:"]
        for i, cand in enumerate(candidates):
            text = cand.chunk.text.replace("\n", " ").strip()
            lines.append(f"[{i}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _parse_scores(text: str, *, n: int) -> dict[int, float] | None:
        match = _JSON_OBJECT.search(text or "")
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            entries = data["scores"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
        out: dict[int, float] = {}
        for entry in entries:
            try:
                idx = int(entry["index"])
                rel = float(entry["relevance"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= idx < n:
                out[idx] = rel
        return out or None
