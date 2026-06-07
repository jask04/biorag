"""Tests for the answer-eval helpers.

The ragas call itself is exercised by the live run. These tests pin down
the deterministic sample selection, the context-passage rebuilder, and
the answer-shape helpers — fast, offline, no Gemini, no langchain.
"""

from __future__ import annotations

from collections.abc import Sequence

from biorag.corpus import Document, QrelEntry, Query
from biorag.eval.answers import (
    AnsweredItem,
    SampleItem,
    answer_question,
    passages_for,
    select_sample,
)
from biorag.generate import Answer
from biorag.retrieve import RetrievalResult


def _query(qid: str, text: str = "x") -> Query:
    return Query(id=qid, text=text)


def _qrel(qid: str, did: str, rel: int) -> QrelEntry:
    return QrelEntry(query_id=qid, doc_id=did, relevance=rel)


# ---------- select_sample ----------


def test_select_sample_takes_first_n_qrel_positive_queries() -> None:
    queries = [_query("Q1"), _query("Q2"), _query("Q3"), _query("Q4")]
    qrels = [
        _qrel("Q1", "D1", 1),
        _qrel("Q2", "D2", 0),  # only zero-relevance — skipped
        _qrel("Q3", "D3", 2),
        _qrel("Q4", "D4", 1),
    ]
    sample = select_sample(queries, qrels, n=2)
    assert [s.query.id for s in sample] == ["Q1", "Q3"]
    assert sample[1].relevant_doc_ids == ["D3"]


def test_select_sample_is_deterministic_and_monotonic() -> None:
    queries = [_query(f"Q{i}") for i in range(1, 6)]
    qrels = [_qrel(f"Q{i}", f"D{i}", 1) for i in range(1, 6)]
    small = select_sample(queries, qrels, n=2)
    large = select_sample(queries, qrels, n=4)
    assert [s.query.id for s in large[:2]] == [s.query.id for s in small]


def test_select_sample_skips_queries_with_no_positive_qrels() -> None:
    queries = [_query("Q1"), _query("Q2"), _query("Q3")]
    qrels = [_qrel("Q1", "D1", 0), _qrel("Q2", "D2", 1)]  # Q3 not in qrels
    sample = select_sample(queries, qrels, n=5)
    assert [s.query.id for s in sample] == ["Q2"]


def test_select_sample_returns_sorted_relevant_doc_ids() -> None:
    queries = [_query("Q1")]
    qrels = [_qrel("Q1", "D9", 1), _qrel("Q1", "D2", 2), _qrel("Q1", "D5", 1)]
    [item] = select_sample(queries, qrels, n=1)
    assert item.relevant_doc_ids == ["D2", "D5", "D9"]


# ---------- passages_for ----------


def _hit(doc_id: str, title: str = "") -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id, score=0.9, chunk_id=f"{doc_id}#0", title=title
    )


def _doc(doc_id: str, title: str, text: str) -> Document:
    return Document(id=doc_id, title=title, text=text)


def test_passages_for_renders_title_and_truncated_text() -> None:
    answer = Answer(
        query="?",
        text="...",
        retrieved=[_hit("D1"), _hit("D2")],
    )
    corpus = {
        "D1": _doc("D1", "Paper one", "Body of paper one."),
        "D2": _doc("D2", "Paper two", "x" * 5000),
    }
    chunks = passages_for(answer, corpus, max_chars=100)
    assert chunks[0].startswith("Paper one. Body")
    assert chunks[1].startswith("Paper two")
    assert chunks[1].endswith("…")


def test_passages_for_falls_back_to_hit_title_when_doc_missing() -> None:
    answer = Answer(query="?", text="...", retrieved=[_hit("D9", title="from-hit")])
    chunks = passages_for(answer, corpus={}, max_chars=100)
    assert chunks == ["from-hit"]


# ---------- answer_question ----------


class _FixedRetriever:
    def __init__(self, hits: list[RetrievalResult]) -> None:
        self._hits = hits

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        return self._hits[:k]


class _FixedGenerator:
    name = "fake-gen"

    def __init__(self, answer: Answer) -> None:
        self._answer = answer

    def answer(self, query: str, hits: Sequence[RetrievalResult]) -> Answer:
        return Answer(
            query=query,
            text=self._answer.text,
            citations=self._answer.citations,
            retrieved=list(hits),
        )


def test_answer_question_flattens_pipeline_output_into_record() -> None:
    item = SampleItem(query=_query("Q1", "is vitamin C effective?"),
                      relevant_doc_ids=["D1"])
    retriever = _FixedRetriever([_hit("D1"), _hit("D2")])
    generator = _FixedGenerator(Answer(query="?", text="Yes [D1].", citations=[]))
    corpus = {
        "D1": _doc("D1", "Vit C", "Body one."),
        "D2": _doc("D2", "Diet", "Body two."),
    }

    result = answer_question(
        item, retriever, generator, corpus, k=2, max_passage_chars=200
    )

    assert isinstance(result, AnsweredItem)
    assert result.query_id == "Q1"
    assert result.retrieved_doc_ids == ["D1", "D2"]
    assert result.relevant_doc_ids == ["D1"]
    assert result.unsupported is False
    assert len(result.contexts) == 2
