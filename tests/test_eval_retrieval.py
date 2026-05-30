"""Tests for the retrieval eval harness.

Every metric is pinned to a closed-form expected value so the harness is
trustworthy at the level that downstream design decisions depend on.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from biorag.corpus import QrelEntry, Query
from biorag.eval.retrieval import (
    NDCG_K,
    RetrieverEvalResult,
    dcg_at_k,
    evaluate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    render_table,
    save_results,
)
from biorag.retrieve import RetrievalResult

# ---------- Recall@k ----------


def test_recall_at_k_full_hit() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_recall_at_k_partial_and_cutoff() -> None:
    # Only "a" is in the top-2; relevant set has two elements.
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == 0.5


def test_recall_at_k_no_overlap() -> None:
    assert recall_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_recall_at_k_rejects_empty_relevant() -> None:
    with pytest.raises(ValueError, match="empty relevant set"):
        recall_at_k(["a"], set(), k=1)


# ---------- Reciprocal Rank ----------


def test_reciprocal_rank_first_hit() -> None:
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_reciprocal_rank_third_hit() -> None:
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_hit() -> None:
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


# ---------- DCG / nDCG ----------


def test_dcg_uses_standard_log2_formula() -> None:
    # rel=3 at rank 1, rel=2 at rank 2, rel=0 at rank 3.
    retrieved = ["a", "b", "c"]
    graded = {"a": 3, "b": 2, "c": 0}
    expected = 3 / math.log2(2) + 2 / math.log2(3)
    assert dcg_at_k(retrieved, graded, k=3) == pytest.approx(expected)


def test_ndcg_perfect_ranking_is_one() -> None:
    graded = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_k(["a", "b", "c"], graded, k=3) == pytest.approx(1.0)


def test_ndcg_reversed_ranking_is_less_than_one_but_positive() -> None:
    graded = {"a": 3, "b": 2, "c": 1}
    score = ndcg_at_k(["c", "b", "a"], graded, k=3)
    assert 0 < score < 1


def test_ndcg_zero_when_no_relevant_doc_retrieved() -> None:
    graded = {"a": 1}
    assert ndcg_at_k(["x", "y"], graded, k=10) == 0.0


def test_ndcg_handles_all_zero_graded() -> None:
    assert ndcg_at_k(["a"], {"a": 0}, k=1) == 0.0


# ---------- evaluate() ----------


class StaticRetriever:
    """Returns a fixed ranking for every query."""

    def __init__(self, ranking: list[str]) -> None:
        self._ranking = ranking

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        return [
            RetrievalResult(doc_id=d, score=1.0 / (i + 1), chunk_id=f"{d}#0", title=d)
            for i, d in enumerate(self._ranking[:k])
        ]


def test_evaluate_averages_metrics_across_scored_queries() -> None:
    queries = [Query(id="Q1", text="t"), Query(id="Q2", text="t")]
    qrels = [
        QrelEntry(query_id="Q1", doc_id="D1", relevance=2),
        QrelEntry(query_id="Q2", doc_id="D3", relevance=1),
    ]
    retriever = StaticRetriever(["D1", "D2", "D3"])

    result = evaluate(retriever, queries, qrels, name="static", k_values=(2,))

    # Q1: D1 found at rank 1 → recall@2=1.0, RR=1.0, nDCG@10 of perfect=1.0
    # Q2: D3 found at rank 3 → recall@2=0.0, RR=1/3, nDCG@10 = (1/log2(4)) / 1
    expected_rr = 0.5 * (1.0 + 1 / 3)
    expected_recall = 0.5 * (1.0 + 0.0)
    expected_ndcg = 0.5 * (1.0 + (1 / math.log2(4)))

    assert result.queries == 2
    assert result.metrics["Recall@2"] == pytest.approx(expected_recall)
    assert result.metrics["MRR"] == pytest.approx(expected_rr)
    assert result.metrics[f"nDCG@{NDCG_K}"] == pytest.approx(expected_ndcg)


def test_evaluate_skips_queries_without_positive_qrels() -> None:
    queries = [Query(id="Q1", text="t"), Query(id="Q2", text="t")]
    # Q1 has only a relevance=0 row; Q2 has none. Both must be skipped.
    qrels = [QrelEntry(query_id="Q1", doc_id="D1", relevance=0)]
    with pytest.raises(RuntimeError, match="no queries had any positive qrels"):
        evaluate(StaticRetriever(["D1"]), queries, qrels, name="x")


# ---------- Persistence + rendering ----------


def test_save_results_round_trips_through_json(tmp_path: Path) -> None:
    results = [
        RetrieverEvalResult(
            name="bm25",
            queries=10,
            metrics={"Recall@10": 0.4, "MRR": 0.3, "nDCG@10": 0.35},
        )
    ]
    out = save_results(results, path=tmp_path / "retrieval.json")
    payload = json.loads(out.read_text())
    assert payload[0]["name"] == "bm25"
    assert payload[0]["metrics"]["Recall@10"] == 0.4


def test_render_table_includes_each_retriever_row() -> None:
    table = render_table(
        [
            RetrieverEvalResult("bm25", 5, {"Recall@10": 0.1, "MRR": 0.2}),
            RetrieverEvalResult("dense", 5, {"Recall@10": 0.3, "MRR": 0.4}),
        ]
    )
    assert "bm25" in table
    assert "dense" in table
    assert "Recall@10" in table
