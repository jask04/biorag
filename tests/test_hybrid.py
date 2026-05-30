"""Tests for Reciprocal Rank Fusion and the hybrid retriever."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from biorag.hybrid import (
    DEFAULT_RRF_K,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from biorag.retrieve import RetrievalResult, Retriever

# ---------- RRF (pure function) ----------


def test_rrf_single_ranking_matches_closed_form() -> None:
    fused = reciprocal_rank_fusion([["A", "B", "C"]], k=60)
    assert fused == [
        ("A", 1.0 / 61),
        ("B", 1.0 / 62),
        ("C", 1.0 / 63),
    ]


def test_rrf_sums_scores_across_rankings() -> None:
    fused = dict(
        reciprocal_rank_fusion([["A", "B"], ["B", "A"]], k=60)
    )
    # A: 1/61 + 1/62, B: 1/62 + 1/61 — identical fused scores.
    assert fused["A"] == pytest.approx(1.0 / 61 + 1.0 / 62)
    assert fused["B"] == pytest.approx(1.0 / 61 + 1.0 / 62)


def test_rrf_promotes_doc_that_appears_in_both_rankings() -> None:
    # "B" is mid-rank in both lists; "A" is top of dense only. RRF should
    # rank B above A because two mid-rank votes beat one top vote at k=60.
    fused = reciprocal_rank_fusion(
        [["A", "B", "C"], ["X", "B", "Y"]], k=60
    )
    order = [doc_id for doc_id, _ in fused]
    assert order.index("B") < order.index("A")


def test_rrf_ignores_empty_rankings_safely() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], ["A"]], k=60) == [("A", 1.0 / 61)]


def test_rrf_tie_break_uses_first_seen_order() -> None:
    # Identical fused scores → preserve the order in which ids first appeared.
    fused = reciprocal_rank_fusion([["X", "Y"], ["Y", "X"]], k=60)
    assert [doc_id for doc_id, _ in fused] == ["X", "Y"]


def test_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="RRF k must be positive"):
        reciprocal_rank_fusion([["A"]], k=0)


def test_rrf_uses_default_k_constant() -> None:
    fused = reciprocal_rank_fusion([["A"]])
    assert fused == [("A", 1.0 / (DEFAULT_RRF_K + 1))]


# ---------- HybridRetriever ----------


def _result(doc_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id, score=score, chunk_id=f"{doc_id}#0", title=f"t-{doc_id}"
    )


class FakeRetriever:
    def __init__(self, hits: list[RetrievalResult]) -> None:
        self._hits = hits
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        self.calls.append((query, k))
        return self._hits[:k]


def test_hybrid_fuses_both_channels_and_returns_top_k() -> None:
    dense = FakeRetriever([_result("A", 0.9), _result("B", 0.7), _result("C", 0.5)])
    sparse = FakeRetriever([_result("B", 11.0), _result("D", 9.0), _result("A", 7.0)])

    hybrid = HybridRetriever(dense, sparse, overfetch=2)
    hits = hybrid.retrieve("q", k=2)

    # Both channels were called with k * overfetch.
    assert dense.calls == [("q", 4)]
    assert sparse.calls == [("q", 4)]
    # B appears at rank 2 in dense and rank 1 in sparse — strongest fusion.
    assert hits[0].doc_id == "B"
    assert len(hits) == 2
    # Hit metadata is carried through from the source retrievers.
    assert all(h.title.startswith("t-") for h in hits)


def test_hybrid_carries_dense_metadata_preferentially() -> None:
    dense = FakeRetriever([
        RetrievalResult("A", 0.9, "A#7", "dense-title-A"),
    ])
    sparse = FakeRetriever([
        RetrievalResult("A", 5.0, "A#0", "sparse-title-A"),
    ])
    [hit] = HybridRetriever(dense, sparse).retrieve("q", k=1)
    assert hit.chunk_id == "A#7"
    assert hit.title == "dense-title-A"


def test_hybrid_satisfies_retriever_protocol() -> None:
    fake: Sequence[RetrievalResult] = []
    h = HybridRetriever(FakeRetriever(list(fake)), FakeRetriever(list(fake)))
    assert isinstance(h, Retriever)


def test_hybrid_rejects_zero_overfetch() -> None:
    with pytest.raises(ValueError, match="overfetch"):
        HybridRetriever(FakeRetriever([]), FakeRetriever([]), overfetch=0)
