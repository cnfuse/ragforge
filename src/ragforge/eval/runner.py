"""Run a QA dataset through the pipeline/agent and produce an :class:`EvalReport`."""

from __future__ import annotations

from statistics import fmean

from ragforge.agent.rag_agent import RagAgent
from ragforge.eval.answer_metrics import expected_match, grounding
from ragforge.eval.dataset import QAExample
from ragforge.eval.report import (
    AnswerResult,
    EvalReport,
    QueryResult,
    RetrievalMetrics,
)
from ragforge.eval.retrieval_metrics import all_metrics
from ragforge.logging import get_logger
from ragforge.pipeline import Pipeline

log = get_logger("eval")


def evaluate_retrieval(pipeline: Pipeline, dataset: list[QAExample], k: int = 5) -> EvalReport:
    """Score retrieval over ``dataset`` and aggregate into a report."""
    rows: list[QueryResult] = []
    for ex in dataset:
        hits = pipeline.retrieve(ex.question, top_k=k)
        ranked = [h.chunk.doc_id for h in hits]
        metrics = all_metrics(ranked, set(ex.relevant_doc_ids), k)
        rows.append(
            QueryResult(id=ex.id, question=ex.question, retrieved_doc_ids=ranked, metrics=metrics)
        )

    agg = _aggregate(rows, k)
    log.info("retrieval eval over %d examples: %s", len(dataset), agg.as_line())
    return EvalReport(num_examples=len(dataset), retrieval=agg, per_query=rows)


def evaluate_answers(
    pipeline: Pipeline, agent: RagAgent, dataset: list[QAExample], k: int = 5
) -> EvalReport:
    """Score both retrieval and generated answers (correctness + grounding)."""
    report = evaluate_retrieval(pipeline, dataset, k)

    answers: list[AnswerResult] = []
    for ex in dataset:
        ans = agent.answer(ex.question)
        answers.append(
            AnswerResult(
                id=ex.id,
                question=ex.question,
                answer=ans.text,
                expected_match=expected_match(ans, ex.expected_substrings),
                grounding=grounding(ans),
            )
        )

    report.answers = answers
    report.mean_expected_match = fmean(a.expected_match for a in answers) if answers else 0.0
    report.mean_grounding = fmean(a.grounding for a in answers) if answers else 0.0
    log.info(
        "answer eval: expected_match=%.3f grounding=%.3f",
        report.mean_expected_match,
        report.mean_grounding,
    )
    return report


def _aggregate(rows: list[QueryResult], k: int) -> RetrievalMetrics:
    if not rows:
        return RetrievalMetrics(k=k)
    return RetrievalMetrics(
        k=k,
        hit_rate=fmean(r.metrics["hit_rate"] for r in rows),
        recall=fmean(r.metrics["recall"] for r in rows),
        mrr=fmean(r.metrics["mrr"] for r in rows),
        ndcg=fmean(r.metrics["ndcg"] for r in rows),
    )
