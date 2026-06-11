"""Structured evaluation reports.

These pydantic models are the typed output of the runner: per-query rows for
drill-down plus dataset-level aggregates for tracking quality over time. They
serialise cleanly to JSON for storage or CI artifacts.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalMetrics(BaseModel):
    """Mean retrieval metrics across a dataset at a fixed cut-off ``k``."""

    k: int
    hit_rate: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0

    def as_line(self) -> str:
        return (
            f"hit_rate@{self.k}={self.hit_rate:.3f}  recall@{self.k}={self.recall:.3f}  "
            f"mrr={self.mrr:.3f}  ndcg@{self.k}={self.ndcg:.3f}"
        )


class QueryResult(BaseModel):
    """Per-query metrics, kept for drill-down and debugging."""

    id: str
    question: str
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class AnswerResult(BaseModel):
    """Per-query answer-quality scores."""

    id: str
    question: str
    answer: str
    expected_match: float = 0.0
    grounding: float = 0.0


class EvalReport(BaseModel):
    """Top-level report combining retrieval aggregates and per-query detail."""

    num_examples: int
    retrieval: RetrievalMetrics
    per_query: list[QueryResult] = Field(default_factory=list)
    answers: list[AnswerResult] = Field(default_factory=list)
    mean_expected_match: float | None = None
    mean_grounding: float | None = None

    def summary(self) -> str:
        lines = [
            f"Evaluated {self.num_examples} example(s)",
            f"  retrieval: {self.retrieval.as_line()}",
        ]
        if self.mean_expected_match is not None:
            lines.append(
                f"  answers:   expected_match={self.mean_expected_match:.3f}  "
                f"grounding={self.mean_grounding:.3f}"
            )
        return "\n".join(lines)
