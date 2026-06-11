"""QA dataset schema and JSONL loader for evaluation.

Each example pairs a question with the set of document ids that should be
retrieved to answer it, plus optional substrings an acceptable answer must
contain. JSONL keeps datasets diffable and append-friendly.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class QAExample(BaseModel):
    """A single labelled evaluation question."""

    id: str
    question: str
    relevant_doc_ids: list[str] = Field(default_factory=list)
    expected_substrings: list[str] = Field(default_factory=list)


def load_dataset(path: str | Path) -> list[QAExample]:
    """Load a JSONL file of :class:`QAExample` records (blank lines ignored)."""
    text = Path(path).read_text(encoding="utf-8")
    examples: list[QAExample] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            examples.append(QAExample(**json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{lineno}: invalid example — {exc}") from exc
    return examples
