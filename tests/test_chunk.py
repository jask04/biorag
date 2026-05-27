"""Tests for the document chunker."""

from __future__ import annotations

import pytest

from biorag.chunk import Chunk, chunk_document, chunk_documents
from biorag.corpus import Document


def _doc(doc_id: str, title: str, text: str) -> Document:
    return Document(id=doc_id, title=title, text=text)


def test_short_doc_returns_single_chunk_with_title_prefixed() -> None:
    doc = _doc("MED-1", "Vitamin C", "Trials show modest effect on cold duration.")
    chunks = chunk_document(doc)
    assert chunks == [
        Chunk(
            id="MED-1#0",
            doc_id="MED-1",
            text="Vitamin C. Trials show modest effect on cold duration.",
        )
    ]


def test_empty_doc_yields_no_chunks() -> None:
    assert chunk_document(_doc("MED-2", "", "")) == []


def test_long_doc_is_windowed_with_overlap_and_keeps_doc_id() -> None:
    body = " ".join(f"w{i}" for i in range(250))
    doc = _doc("MED-3", "", body)
    chunks = chunk_document(doc, max_words=100, overlap=20)

    assert [c.doc_id for c in chunks] == ["MED-3", "MED-3", "MED-3"]
    assert [c.id for c in chunks] == ["MED-3#0", "MED-3#1", "MED-3#2"]
    # First chunk is 100 words; stride is 100-20=80, so chunk[1] starts at w80.
    assert chunks[0].text.split()[:3] == ["w0", "w1", "w2"]
    assert chunks[1].text.split()[:3] == ["w80", "w81", "w82"]
    # Overlap region appears in both adjacent chunks.
    overlap_tokens = set(chunks[0].text.split()[-20:])
    assert overlap_tokens.issubset(set(chunks[1].text.split()))


def test_chunk_documents_streams_across_inputs() -> None:
    docs = [_doc("A", "ta", "xa"), _doc("B", "tb", "xb")]
    ids = [c.id for c in chunk_documents(docs)]
    assert ids == ["A#0", "B#0"]


@pytest.mark.parametrize(
    ("max_words", "overlap"),
    [(0, 0), (-1, 0), (10, 10), (10, 11), (10, -1)],
)
def test_invalid_parameters_raise(max_words: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_document(_doc("X", "t", "x"), max_words=max_words, overlap=overlap)
