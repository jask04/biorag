"""Dense retrieval over a Qdrant collection.

:class:`DenseRetriever` embeds a query and returns the top-k *documents*.
Because a document can be split into several chunks, retrieval over-fetches
chunk hits and deduplicates to documents (keeping each document's best
hit), so callers always get distinct doc ids — the unit the Day 6 eval
harness scores against.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from biorag.embed import Embedder
from biorag.index import collection_name, default_client

DEFAULT_OVERFETCH: Final[int] = 5


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A retrieved document with the score of its best-matching chunk."""

    doc_id: str
    score: float
    chunk_id: str
    title: str


def dedup_to_documents(
    points: Iterable[ScoredPoint],
    k: int,
) -> list[RetrievalResult]:
    """Collapse chunk-level hits (descending score) to top-k documents."""
    results: list[RetrievalResult] = []
    seen: set[str] = set()
    for point in points:
        payload = point.payload or {}
        doc_id = str(payload.get("doc_id", ""))
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        results.append(
            RetrievalResult(
                doc_id=doc_id,
                score=float(point.score),
                chunk_id=str(payload.get("chunk_id", "")),
                title=str(payload.get("title", "")),
            )
        )
        if len(results) >= k:
            break
    return results


class DenseRetriever:
    """Embed the query and ANN-search a Qdrant collection."""

    def __init__(
        self,
        embedder: Embedder,
        client: QdrantClient | None = None,
        collection: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.client = client if client is not None else default_client()
        self.collection = collection or collection_name(embedder.name)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        overfetch: int = DEFAULT_OVERFETCH,
    ) -> list[RetrievalResult]:
        query_vector = self.embedder.embed([query])[0].tolist()
        response = self.client.query_points(
            self.collection,
            query=query_vector,
            limit=k * overfetch,
            with_payload=True,
        )
        return dedup_to_documents(response.points, k)


__all__ = [
    "DEFAULT_OVERFETCH",
    "DenseRetriever",
    "RetrievalResult",
    "dedup_to_documents",
]
