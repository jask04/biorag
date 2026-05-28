"""Tests for chunk-to-document deduplication in the retriever.

``ScoredPoint`` instances are constructed directly so we exercise the real
dedup logic against the same type Qdrant returns, without a live client.
"""

from __future__ import annotations

from qdrant_client.models import ScoredPoint

from biorag.retrieve import RetrievalResult, dedup_to_documents


def _point(score: float, doc_id: str, chunk_id: str, title: str = "t") -> ScoredPoint:
    return ScoredPoint(
        id=chunk_id,
        version=0,
        score=score,
        payload={"doc_id": doc_id, "chunk_id": chunk_id, "title": title},
    )


def test_keeps_best_chunk_per_document_in_order() -> None:
    points = [
        _point(0.91, "D1", "D1#0", "Doc one"),
        _point(0.88, "D2", "D2#0", "Doc two"),
        _point(0.80, "D1", "D1#1", "Doc one"),  # lower-scoring dup of D1
        _point(0.75, "D3", "D3#0", "Doc three"),
    ]
    results = dedup_to_documents(points, k=10)
    assert results == [
        RetrievalResult(doc_id="D1", score=0.91, chunk_id="D1#0", title="Doc one"),
        RetrievalResult(doc_id="D2", score=0.88, chunk_id="D2#0", title="Doc two"),
        RetrievalResult(doc_id="D3", score=0.75, chunk_id="D3#0", title="Doc three"),
    ]


def test_respects_k_after_dedup() -> None:
    points = [
        _point(0.9, "D1", "D1#0"),
        _point(0.8, "D1", "D1#1"),
        _point(0.7, "D2", "D2#0"),
        _point(0.6, "D3", "D3#0"),
    ]
    results = dedup_to_documents(points, k=2)
    assert [r.doc_id for r in results] == ["D1", "D2"]


def test_skips_points_without_doc_id() -> None:
    points = [
        ScoredPoint(id="x", version=0, score=0.9, payload=None),
        ScoredPoint(id="y", version=0, score=0.8, payload={}),
        _point(0.7, "D9", "D9#0"),
    ]
    results = dedup_to_documents(points, k=5)
    assert [r.doc_id for r in results] == ["D9"]
