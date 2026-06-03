"""Tests for grounded answer generation.

The real :class:`GeminiAnswerGenerator` is exercised by the ``ask`` CLI
smoke and by Day 10's ragas eval. These tests pin down the pure helpers
(passage formatting, citation extraction, citation building) and the
``Answer`` dataclass contract — fast and offline.
"""

from __future__ import annotations

from biorag.corpus import Document
from biorag.generate import (
    UNSUPPORTED_MARKER,
    Answer,
    Citation,
    build_citations,
    extract_citations,
    format_passages,
)
from biorag.retrieve import RetrievalResult


def _hit(doc_id: str, title: str = "", score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id, score=score, chunk_id=f"{doc_id}#0", title=title
    )


def _doc(doc_id: str, title: str, text: str) -> Document:
    return Document(id=doc_id, title=title, text=text)


# ---------- format_passages ----------


def test_format_passages_includes_title_and_text_per_doc() -> None:
    hits = [_hit("MED-1"), _hit("MED-2")]
    corpus = {
        "MED-1": _doc("MED-1", "Vitamin C trials", "Body of paper one."),
        "MED-2": _doc("MED-2", "Mediterranean diet", "Body of paper two."),
    }
    block = format_passages(hits, corpus)
    assert "[MED-1] Vitamin C trials" in block
    assert "Body of paper one." in block
    assert "[MED-2] Mediterranean diet" in block
    assert "Body of paper two." in block


def test_format_passages_truncates_long_text() -> None:
    long_body = "x" * 5000
    hits = [_hit("D1")]
    corpus = {"D1": _doc("D1", "t", long_body)}
    block = format_passages(hits, corpus, max_chars=100)
    assert block.endswith("…")
    # Body section after the title line is at most max_chars + the
    # ellipsis sentinel, comfortably under the original 5000 chars.
    assert len(block) < 500


def test_format_passages_falls_back_to_retrieval_title_when_doc_missing() -> None:
    hits = [_hit("ORPHAN", title="from-retrieval")]
    block = format_passages(hits, corpus={})
    assert "[ORPHAN] from-retrieval" in block


# ---------- extract_citations ----------


def test_extract_citations_handles_single_and_grouped_brackets() -> None:
    text = (
        "Vitamin C reduces cold duration [MED-123]. Other trials disagree "
        "[MED-456, MED-789]. A repeat reference to [MED-123] is collapsed."
    )
    assert extract_citations(text, ["MED-123", "MED-456", "MED-789"]) == [
        "MED-123",
        "MED-456",
        "MED-789",
    ]


def test_extract_citations_filters_to_allowed_ids() -> None:
    # The model hallucinated MED-999 — must NOT appear in structured output.
    text = "Strong effect [MED-1, MED-999]. Also [MED-2]."
    assert extract_citations(text, ["MED-1", "MED-2"]) == ["MED-1", "MED-2"]


def test_extract_citations_returns_first_appearance_order() -> None:
    text = "[B-2] then [A-1] then [B-2] again then [C-3]."
    assert extract_citations(text, ["A-1", "B-2", "C-3"]) == ["B-2", "A-1", "C-3"]


def test_extract_citations_returns_empty_when_no_brackets() -> None:
    assert extract_citations("a sentence with no citations", ["MED-1"]) == []


# ---------- build_citations ----------


def test_build_citations_prefers_corpus_title_then_falls_back_to_hit_title() -> None:
    corpus = {"MED-1": _doc("MED-1", "Corpus title", "x")}
    hits = [_hit("MED-1", title="hit-title"), _hit("MED-2", title="hit-title-2")]
    cites = build_citations(["MED-1", "MED-2"], corpus, hits)
    assert cites == [
        Citation(doc_id="MED-1", title="Corpus title"),
        Citation(doc_id="MED-2", title="hit-title-2"),
    ]


# ---------- Answer ----------


def test_answer_unsupported_is_true_when_text_matches_marker() -> None:
    answer = Answer(query="?", text=UNSUPPORTED_MARKER)
    assert answer.unsupported is True


def test_answer_unsupported_is_false_for_a_real_answer() -> None:
    answer = Answer(
        query="?",
        text="Vitamin C reduces cold duration [MED-1].",
        citations=[Citation("MED-1", "t")],
    )
    assert answer.unsupported is False
