"""Compare retrieval configurations on the same dataset (an ablation harness).

Each named configuration is a full :class:`Settings`; the corpus is indexed under
each and scored with the retrieval metrics *and* timed, so the impact of toggling
hybrid retrieval, reranking, or MMR is measured on both quality and cost rather
than assumed. Returns per-config results and renders a comparison table.
"""

from __future__ import annotations

import time

from pydantic import BaseModel

from ragforge.config import Settings
from ragforge.eval.dataset import QAExample
from ragforge.eval.report import RetrievalMetrics
from ragforge.eval.runner import evaluate_retrieval
from ragforge.logging import get_logger
from ragforge.pipeline import Pipeline
from ragforge.types import Document

log = get_logger("eval.compare")


class ConfigResult(BaseModel):
    """Quality metrics plus mean query latency for one configuration."""

    metrics: RetrievalMetrics
    mean_latency_ms: float


def default_matrix() -> dict[str, Settings]:
    """A standard ablation matrix over the retrieval stages."""
    return {
        "dense": Settings(),
        "hybrid": Settings(hybrid_enabled=True),
        "hybrid+rerank": Settings(
            hybrid_enabled=True, rerank_enabled=True, rerank_provider="lexical"
        ),
        "hybrid+mmr": Settings(hybrid_enabled=True, mmr_enabled=True),
    }


def compare_retrieval_configs(
    documents: list[Document],
    dataset: list[QAExample],
    configs: dict[str, Settings],
    *,
    k: int = 5,
) -> dict[str, ConfigResult]:
    """Index ``documents`` and score + time retrieval under each named config."""
    results: dict[str, ConfigResult] = {}
    for name, settings in configs.items():
        pipeline = Pipeline(settings)
        pipeline.ingest(documents)

        start = time.perf_counter()
        for ex in dataset:
            pipeline.retrieve(ex.question, top_k=k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        mean_latency = elapsed_ms / len(dataset) if dataset else 0.0

        report = evaluate_retrieval(pipeline, dataset, k=k)
        results[name] = ConfigResult(metrics=report.retrieval, mean_latency_ms=mean_latency)
        log.info(
            "config %-16s %s  %.2f ms/query",
            name,
            report.retrieval.as_line(),
            mean_latency,
        )
    return results


def format_comparison(results: dict[str, ConfigResult]) -> str:
    """Render the comparison as a fixed-width table, best config marked."""
    if not results:
        return "(no configurations to compare)"
    k = next(iter(results.values())).metrics.k
    header = (
        f"{'config':<18}{'hit_rate':>10}{'recall':>9}{'mrr':>8}{'ndcg':>8}{'ms/query':>11}"
    )
    rows = [header, "-" * len(header)]
    best_name = max(results, key=lambda n: results[n].metrics.ndcg)
    for name, res in results.items():
        m = res.metrics
        mark = "  *" if name == best_name else ""
        rows.append(
            f"{name:<18}{m.hit_rate:>10.3f}{m.recall:>9.3f}{m.mrr:>8.3f}"
            f"{m.ndcg:>8.3f}{res.mean_latency_ms:>11.2f}{mark}"
        )
    rows.append(f"(metrics @k={k}; * = best by nDCG)")
    return "\n".join(rows)
