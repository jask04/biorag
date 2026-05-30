"""Tests for the BM25 retriever and its tokenizer."""

from __future__ import annotations

import pytest

from biorag.bm25 import BM25Retriever, tokenize
from biorag.corpus import Document
from biorag.retrieve import Retriever


def _doc(doc_id: str, title: str, text: str) -> Document:
    return Document(id=doc_id, title=title, text=text)


def test_tokenize_lowercases_and_splits_on_non_word() -> None:
    assert tokenize("Vitamin-C: 1000mg!") == ["vitamin", "c", "1000mg"]
    assert tokenize("") == []


def test_bm25_ranks_topic_match_above_unrelated() -> None:
    docs = [
        _doc("D1", "Vitamin C and the common cold",
             "ascorbic acid trials for upper respiratory infections"),
        _doc("D2", "Mediterranean diet outcomes",
             "olive oil and cardiovascular endpoints"),
        _doc("D3", "Curcumin bioavailability",
             "turmeric pharmacokinetics"),
    ]
    retriever = BM25Retriever(docs)
    hits = retriever.retrieve("vitamin C cold trial", k=3)
    assert hits[0].doc_id == "D1"
    assert hits[0].score > 0.0
    assert {h.doc_id for h in hits}.issubset({"D1", "D2", "D3"})


def test_bm25_respects_k_and_drops_zero_score_docs() -> None:
    # Five docs so BM25Okapi's IDF doesn't collapse to zero, with only one
    # doc carrying the query term: k=10 must yield exactly that one hit.
    docs = [
        _doc("D1", "vitamin c", "ascorbic acid"),
        _doc("D2", "diet", "weight loss"),
        _doc("D3", "exercise", "cardio"),
        _doc("D4", "sleep", "circadian rhythm"),
        _doc("D5", "mindfulness", "meditation"),
    ]
    hits = BM25Retriever(docs).retrieve("vitamin", k=10)
    assert [h.doc_id for h in hits] == ["D1"]


def test_empty_query_returns_no_hits() -> None:
    docs = [_doc("D1", "t", "x")]
    assert BM25Retriever(docs).retrieve("!!!", k=5) == []


def test_satisfies_retriever_protocol() -> None:
    docs = [_doc("D1", "t", "x")]
    assert isinstance(BM25Retriever(docs), Retriever)


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        BM25Retriever([])
