"""Offline retrieval evaluation against BEIR-style gold qrels.

This module is the **spine** of biorag: every later technique (rerank,
HyDE, domain embeddings) earns its place by moving these numbers.

Metrics implemented from first principles (one test pins each closed-form):

- **Recall@k** — binary; fraction of relevant docs found in the top-k.
- **MRR** — reciprocal rank of the first relevant doc (0 if none).
- **nDCG@10** — graded; ``rel_i / log2(i+1)`` summed over the top-k, divided
  by the ideal DCG over the same number of positions.

Queries with no positive qrel are skipped (recall is undefined). The
"positive" filter is ``relevance > 0`` so the harness works against any
graded qrel set, not just binary ones.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

from biorag.corpus import QrelEntry, Query
from biorag.retrieve import Retriever

DEFAULT_K_VALUES: Final[tuple[int, ...]] = (10, 20)
NDCG_K: Final[int] = 10
DEFAULT_RESULTS_DIR: Final[Path] = Path("eval_results")


# ---------- Pure metric functions ----------


def recall_at_k(
    retrieved: Sequence[str], relevant: set[str], k: int
) -> float:
    """``|retrieved[:k] ∩ relevant| / |relevant|``."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("recall is undefined for an empty relevant set")
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """``1 / rank`` of the first relevant doc, or 0 if none."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(
    retrieved: Sequence[str], graded: Mapping[str, int], k: int
) -> float:
    """Linear-gain Discounted Cumulative Gain over the top-k.

    Uses ``rel_i / log2(i + 1)`` with 1-based rank — the standard formula
    that makes ``log2(2) = 1`` at position 1.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    total = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        rel = graded.get(doc_id, 0)
        if rel > 0:
            total += rel / math.log2(rank + 1)
    return total


def ndcg_at_k(
    retrieved: Sequence[str], graded: Mapping[str, int], k: int
) -> float:
    """DCG@k divided by the ideal DCG over the same window."""
    if k <= 0:
        raise ValueError("k must be positive")
    ideal_scores = sorted(graded.values(), reverse=True)[:k]
    idcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(ideal_scores, start=1)
        if rel > 0
    )
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, graded, k) / idcg


# ---------- Aggregated eval ----------


@dataclass(frozen=True, slots=True)
class RetrieverEvalResult:
    """Mean metrics for one retriever over the scored query set."""

    name: str
    queries: int
    metrics: dict[str, float] = field(default_factory=dict)


def _group_qrels(qrels: Sequence[QrelEntry]) -> dict[str, dict[str, int]]:
    """``query_id -> {doc_id: relevance}`` lookup."""
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for entry in qrels:
        grouped[entry.query_id][entry.doc_id] = entry.relevance
    return grouped


def evaluate(
    retriever: Retriever,
    queries: Sequence[Query],
    qrels: Sequence[QrelEntry],
    name: str,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> RetrieverEvalResult:
    """Run ``retriever`` over scored queries and average the metrics.

    A query is "scored" iff it has at least one ``relevance > 0`` qrel.
    All other queries are skipped silently — they cannot contribute to
    recall and would dilute MRR with undefined values.
    """
    if not k_values:
        raise ValueError("k_values must not be empty")
    by_qid = _group_qrels(qrels)
    max_k = max(max(k_values), NDCG_K)

    recall_sums: dict[int, float] = dict.fromkeys(k_values, 0.0)
    mrr_sum = 0.0
    ndcg_sum = 0.0
    counted = 0

    for query in queries:
        graded = by_qid.get(query.id)
        if not graded:
            continue
        relevant = {doc_id for doc_id, rel in graded.items() if rel > 0}
        if not relevant:
            continue

        hits = retriever.retrieve(query.text, k=max_k)
        retrieved_ids = [hit.doc_id for hit in hits]

        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved_ids, relevant, k)
        mrr_sum += reciprocal_rank(retrieved_ids, relevant)
        ndcg_sum += ndcg_at_k(retrieved_ids, graded, NDCG_K)
        counted += 1

    if counted == 0:
        raise RuntimeError("no queries had any positive qrels — nothing to score")

    metrics: dict[str, float] = {
        f"Recall@{k}": recall_sums[k] / counted for k in k_values
    }
    metrics["MRR"] = mrr_sum / counted
    metrics[f"nDCG@{NDCG_K}"] = ndcg_sum / counted

    return RetrieverEvalResult(name=name, queries=counted, metrics=metrics)


# ---------- Persistence + rendering ----------


def render_table(results: Sequence[RetrieverEvalResult]) -> str:
    """Render a fixed-width Markdown-ish table of the results."""
    if not results:
        return "(no results)"
    metric_keys = list(results[0].metrics)
    name_w = max(len("retriever"), max(len(r.name) for r in results))
    metric_w = max(8, *(len(k) for k in metric_keys))

    def row(label: str, cells: Sequence[str]) -> str:
        return (
            f"{label:<{name_w}}  N={cells[0]:>5}  "
            + "  ".join(f"{c:>{metric_w}}" for c in cells[1:])
        )

    lines = [
        f"{'retriever':<{name_w}}  {'N':>7}  "
        + "  ".join(f"{k:>{metric_w}}" for k in metric_keys),
        "-" * (name_w + 2 + 7 + 2 + (metric_w + 2) * len(metric_keys)),
    ]
    for r in results:
        cells = [str(r.queries)] + [f"{r.metrics[k]:.4f}" for k in metric_keys]
        lines.append(row(r.name, cells))
    return "\n".join(lines)


def save_results(
    results: Sequence[RetrieverEvalResult],
    path: Path = DEFAULT_RESULTS_DIR / "retrieval.json",
) -> Path:
    """Write results as JSON; returns the resolved path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_K_VALUES",
    "DEFAULT_RESULTS_DIR",
    "NDCG_K",
    "RetrieverEvalResult",
    "dcg_at_k",
    "evaluate",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "render_table",
    "save_results",
]
