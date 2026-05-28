"""Tests for the pure helpers in the indexing layer.

Network-touching paths (``QdrantIndex`` upsert, ``DenseRetriever``) are
covered by the live smoke build; here we pin the deterministic, offline
pieces: collection naming and point construction.
"""

from __future__ import annotations

from biorag.chunk import Chunk
from biorag.index import build_point, collection_name, point_id


def test_collection_name_is_qdrant_safe() -> None:
    assert collection_name("BAAI/bge-small-en-v1.5") == (
        "biorag_BAAI__bge_small_en_v1_5"
    )
    name = collection_name("NeuML/pubmedbert-base-embeddings")
    assert name == "biorag_NeuML__pubmedbert_base_embeddings"
    assert "/" not in name and "-" not in name and "." not in name


def test_point_id_is_deterministic_uuid() -> None:
    first = point_id("MED-1#0")
    assert first == point_id("MED-1#0")
    assert first != point_id("MED-1#1")
    # Looks like a UUID (5 dash-separated hex groups).
    assert len(first.split("-")) == 5


def test_build_point_carries_provenance_payload() -> None:
    chunk = Chunk(id="MED-7#2", doc_id="MED-7", text="some abstract text")
    point = build_point(chunk, [0.1, 0.2, 0.3], title="A study title")

    assert point.id == point_id("MED-7#2")
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.payload == {
        "doc_id": "MED-7",
        "chunk_id": "MED-7#2",
        "title": "A study title",
    }
