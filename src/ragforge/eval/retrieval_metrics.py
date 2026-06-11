"""Standard information-retrieval metrics over a single ranked result list.

Each function takes ``ranked`` — document ids ordered best-first — and
``relevant`` — the set of ids that count as correct — and returns a score in
``[0, 1]``. These are the building blocks the runner averages across a dataset.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _dedupe_preserve_order(ranked: Iterable[str]) -> list[str]:
    """Collapse repeated doc ids (a doc may yield several chunks) to first hit."""
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in ranked:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant doc appears in the top-k, else 0.0."""
    if not relevant:
        return 0.0
    return 1.0 if any(d in relevant for d in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant docs that appear in the top-k."""
    if not relevant:
        return 0.0
    found = sum(1 for d in set(ranked[:k]) if d in relevant)
    return found / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant doc (0 if none retrieved)."""
    if not relevant:
        return 0.0
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance normalised discounted cumulative gain at k."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def all_metrics(ranked: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    """Compute every retrieval metric for one query, deduping by document."""
    deduped = _dedupe_preserve_order(ranked)
    return {
        "hit_rate": hit_at_k(deduped, relevant, k),
        "recall": recall_at_k(deduped, relevant, k),
        "mrr": reciprocal_rank(deduped, relevant),
        "ndcg": ndcg_at_k(deduped, relevant, k),
    }
