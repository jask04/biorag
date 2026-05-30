"""Hybrid retrieval: dense + sparse fused with Reciprocal Rank Fusion.

RRF (Cormack et al., 2009) is a rank-only fuser: it ignores per-retriever
scores (which live on different scales for cosine vs BM25) and combines
positions. The score for a document ``d`` is the sum over the contributing
retrievers ``r`` of ``1 / (k + rank_r(d))`` where ``rank_r(d)`` is 1-based.
``k`` damps the head — the default of 60 follows the original paper.

The hybrid retriever calls both channels with an over-fetch factor so the
fusion has more candidates to work with than the final top-k.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from biorag.retrieve import RetrievalResult, Retriever

DEFAULT_RRF_K: Final[int] = 60
DEFAULT_OVERFETCH: Final[int] = 5


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into a single score-sorted list.

    Args:
        rankings: Each entry is a list of doc ids in descending relevance
            order (rank 1 first). Different rankings may overlap or not.
        k: Damping constant; larger ``k`` flattens the contribution of the
            very top ranks. The literature default is 60.

    Returns:
        ``[(doc_id, fused_score), ...]`` sorted by score descending. Ties
        are broken by first-seen order across the input rankings, which
        gives a deterministic output for any fixed inputs.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")

    scores: dict[str, float] = {}
    seen_order: dict[str, int] = {}
    counter = 0
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in seen_order:
                seen_order[doc_id] = counter
                counter += 1

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], seen_order[item[0]]),
    )


class HybridRetriever:
    """Dense ANN + BM25 fused by RRF, exposing the :class:`Retriever` shape."""

    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        rrf_k: int = DEFAULT_RRF_K,
        overfetch: int = DEFAULT_OVERFETCH,
    ) -> None:
        if overfetch < 1:
            raise ValueError("overfetch must be >= 1")
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k
        self._overfetch = overfetch

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        pool_size = k * self._overfetch
        dense_hits = self._dense.retrieve(query, k=pool_size)
        sparse_hits = self._sparse.retrieve(query, k=pool_size)

        # Keep one representative hit per doc id for citation metadata
        # (title, best chunk id). Dense gets first pick because its chunk-
        # level granularity is more informative for citations downstream.
        meta: dict[str, RetrievalResult] = {}
        for hit in dense_hits:
            meta.setdefault(hit.doc_id, hit)
        for hit in sparse_hits:
            meta.setdefault(hit.doc_id, hit)

        fused = reciprocal_rank_fusion(
            [
                [h.doc_id for h in dense_hits],
                [h.doc_id for h in sparse_hits],
            ],
            k=self._rrf_k,
        )

        return [
            RetrievalResult(
                doc_id=doc_id,
                score=score,
                chunk_id=meta[doc_id].chunk_id,
                title=meta[doc_id].title,
            )
            for doc_id, score in fused[:k]
        ]


__all__ = [
    "DEFAULT_OVERFETCH",
    "DEFAULT_RRF_K",
    "HybridRetriever",
    "reciprocal_rank_fusion",
]
