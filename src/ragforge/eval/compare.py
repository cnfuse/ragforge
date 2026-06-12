"""Compare retrieval configurations on the same dataset (an ablation harness).

Each named configuration is a full :class:`Settings`; the corpus is indexed under
each and scored with the retrieval metrics, so the impact of toggling hybrid
retrieval, reranking, or MMR is measured rather than assumed. Returns the
per-config aggregate metrics and renders a comparison table.
"""

from __future__ import annotations

from ragforge.config import Settings
from ragforge.eval.dataset import QAExample
from ragforge.eval.report import RetrievalMetrics
from ragforge.eval.runner import evaluate_retrieval
from ragforge.logging import get_logger
from ragforge.pipeline import Pipeline
from ragforge.types import Document

log = get_logger("eval.compare")


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
) -> dict[str, RetrievalMetrics]:
    """Index ``documents`` and score retrieval under each named configuration."""
    results: dict[str, RetrievalMetrics] = {}
    for name, settings in configs.items():
        pipeline = Pipeline(settings)
        pipeline.ingest(documents)
        report = evaluate_retrieval(pipeline, dataset, k=k)
        results[name] = report.retrieval
        log.info("config %-16s %s", name, report.retrieval.as_line())
    return results


def format_comparison(results: dict[str, RetrievalMetrics]) -> str:
    """Render the comparison as a fixed-width table, best config marked."""
    if not results:
        return "(no configurations to compare)"
    k = next(iter(results.values())).k
    header = f"{'config':<18}{'hit_rate':>10}{'recall':>9}{'mrr':>8}{'ndcg':>8}"
    rows = [header, "-" * len(header)]
    best_name = max(results, key=lambda n: results[n].ndcg)
    for name, m in results.items():
        mark = "  *" if name == best_name else ""
        rows.append(
            f"{name:<18}{m.hit_rate:>10.3f}{m.recall:>9.3f}{m.mrr:>8.3f}{m.ndcg:>8.3f}{mark}"
        )
    rows.append(f"(metrics @k={k}; * = best by nDCG)")
    return "\n".join(rows)
