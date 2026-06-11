"""Answer-quality checks: did the agent get it right, and is it grounded?

``expected_match`` is a correctness proxy — does the answer contain the
substrings a good answer should. ``grounding`` is a faithfulness proxy — how much
of the answer's salient vocabulary is actually supported by the passages it
cited, which flags answers that drift beyond their evidence.
"""

from __future__ import annotations

import re

from ragforge.types import Answer

_WORD = re.compile(r"[a-z0-9]{3,}")
# Common function words plus a few RAG-prompt artefacts, excluded so grounding
# measures overlap of *content* vocabulary rather than filler.
_STOP = frozenset(
    {
        "the", "and", "but", "for", "nor", "of", "to", "in", "on", "at", "by",
        "with", "from", "into", "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "its", "as", "not", "can", "will",
        "would", "should", "could", "based", "context", "retrieved",
    }
)


def expected_match(answer: Answer, expected_substrings: list[str]) -> float:
    """Fraction of expected substrings present in the answer (case-insensitive)."""
    if not expected_substrings:
        return 1.0  # nothing to require -> trivially satisfied
    text = answer.text.lower()
    hits = sum(1 for s in expected_substrings if s.lower() in text)
    return hits / len(expected_substrings)


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def grounding(answer: Answer) -> float:
    """Share of the answer's content words found in its cited passages.

    Returns 1.0 for an answer with no content words (nothing to ground) and
    0.0 when the answer makes claims but cites nothing.
    """
    answer_words = _content_words(answer.text)
    if not answer_words:
        return 1.0
    if not answer.citations:
        return 0.0
    cited_words: set[str] = set()
    for sc in answer.citations:
        cited_words |= _content_words(sc.chunk.text)
    supported = sum(1 for w in answer_words if w in cited_words)
    return supported / len(answer_words)
