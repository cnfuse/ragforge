"""Evaluation harness: measure retrieval quality and answer grounding.

Changes to chunking, embeddings, or prompts should be judged against numbers,
not vibes. This package provides a small QA dataset format, standard
information-retrieval metrics (hit-rate, recall@k, MRR, nDCG), answer-grounding
checks, and a runner that produces a structured report.
"""

from __future__ import annotations

from ragforge.eval.dataset import QAExample, load_dataset
from ragforge.eval.report import EvalReport, RetrievalMetrics
from ragforge.eval.runner import evaluate_answers, evaluate_retrieval

__all__ = [
    "QAExample",
    "load_dataset",
    "EvalReport",
    "RetrievalMetrics",
    "evaluate_retrieval",
    "evaluate_answers",
]
