"""Build and populate Qdrant collections for dense retrieval.

One collection per embedder (the general-vs-biomedical comparison needs
both side by side). Each chunk becomes a point whose payload carries the
provenance needed for citations and evaluation: ``doc_id``, ``chunk_id``,
and ``title``. Vectors are stored with cosine distance since the embedder
already L2-normalizes them.

The Qdrant API key is read from settings only inside :func:`default_client`
and is never logged.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import batched
from typing import Final
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from biorag.chunk import (
    DEFAULT_MAX_WORDS,
    DEFAULT_OVERLAP,
    Chunk,
    chunk_documents,
)
from biorag.config import get_settings
from biorag.corpus import Document
from biorag.embed import Embedder

COLLECTION_PREFIX: Final[str] = "biorag"
DEFAULT_UPSERT_BATCH: Final[int] = 128


def collection_name(model_name: str) -> str:
    """Derive a stable, Qdrant-safe collection name from a model id."""
    slug = model_name.replace("/", "__").replace("-", "_").replace(".", "_")
    return f"{COLLECTION_PREFIX}_{slug}"


def default_client() -> QdrantClient:
    """Construct a Qdrant Cloud client from settings.

    The URL and API key live in ``.env`` / the deploy secret store. The key
    is passed straight to the client and never logged.
    """
    settings = get_settings()
    if not settings.qdrant_url or not settings.qdrant_api_key:
        raise RuntimeError(
            "QDRANT_URL and QDRANT_API_KEY must be set in .env to reach Qdrant"
        )
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
    )


def point_id(chunk_id: str) -> str:
    """Map a chunk id (e.g. ``MED-1#0``) to a deterministic UUID point id."""
    return str(uuid5(NAMESPACE_URL, chunk_id))


def build_point(chunk: Chunk, vector: Sequence[float], title: str) -> PointStruct:
    """Build a Qdrant point with citation/eval payload for one chunk."""
    return PointStruct(
        id=point_id(chunk.id),
        vector=list(vector),
        payload={
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.id,
            "title": title,
        },
    )


class QdrantIndex:
    """Owns a single collection and knows how to (re)build it from documents."""

    def __init__(
        self,
        embedder: Embedder,
        client: QdrantClient | None = None,
        collection: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.client = client if client is not None else default_client()
        self.collection = collection or collection_name(embedder.name)

    def recreate(self) -> None:
        """Drop and recreate the collection with the embedder's dimension."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(
                size=self.embedder.dimension,
                distance=Distance.COSINE,
            ),
        )

    def index_documents(
        self,
        docs: Sequence[Document],
        max_words: int = DEFAULT_MAX_WORDS,
        overlap: int = DEFAULT_OVERLAP,
        batch_size: int = DEFAULT_UPSERT_BATCH,
    ) -> int:
        """Chunk, embed, and upsert documents. Returns the chunk count."""
        titles = {doc.id: doc.title for doc in docs}
        chunks = list(chunk_documents(docs, max_words=max_words, overlap=overlap))
        total = 0
        for batch in batched(chunks, batch_size):
            vectors = self.embedder.embed([c.text for c in batch])
            points = [
                build_point(chunk, vectors[i].tolist(), titles.get(chunk.doc_id, ""))
                for i, chunk in enumerate(batch)
            ]
            self.client.upsert(self.collection, points=points)
            total += len(points)
        return total


__all__ = [
    "COLLECTION_PREFIX",
    "DEFAULT_UPSERT_BATCH",
    "QdrantIndex",
    "build_point",
    "collection_name",
    "default_client",
    "point_id",
]
