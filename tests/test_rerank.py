"""Tests for the cross-encoder rerank stage.

The real :class:`CrossEncoderReranker` (sentence-transformers + torch) is
exercised by the eval harness when it's run with rerank configs. These
tests pin down the composition logic with a deterministic fake reranker,
so they stay fast and offline.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from biorag.corpus import Document
from biorag.rerank import (
    DEFAULT_POOL_SIZE,
    Reranker,
    RerankingRetriever,
)
from biorag.retrieve import RetrievalResult


class ScriptedReranker:
    """Returns scores from a ``{passage_substring: score}`` lookup."""

    def __init__(self, table: dict[str, float]) -> None:
        self.name = "scripted"
        self._table = table
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        scores: list[float] = []
        for passage in passages:
            score = 0.0
            for key, val in self._table.items():
                if key in passage:
                    score = val
                    break
            scores.append(score)
        return scores


class FakeBase:
    def __init__(self, hits: list[RetrievalResult]) -> None:
        self._hits = hits
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        self.calls.append((query, k))
        return self._hits[:k]


def _hit(doc_id: str, base_score: float) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id,
        score=base_score,
        chunk_id=f"{doc_id}#0",
        title=f"title-{doc_id}",
    )


def _corpus(*docs: Document) -> dict[str, Document]:
    return {d.id: d for d in docs}


def test_scripted_reranker_satisfies_protocol() -> None:
    assert isinstance(ScriptedReranker({}), Reranker)


def test_reranker_reorders_base_hits_by_score() -> None:
    base = FakeBase([_hit("A", 0.9), _hit("B", 0.8), _hit("C", 0.7)])
    corpus = _corpus(
        Document(id="A", title="title-A", text="apples and oranges"),
        Document(id="B", title="title-B", text="ascorbic acid and citrus"),
        Document(id="C", title="title-C", text="bananas potassium"),
    )
    # The reranker promotes B above the base order.
    reranker = ScriptedReranker({"ascorbic": 5.0, "apples": 2.0, "bananas": 1.0})

    out = RerankingRetriever(base, reranker, corpus, pool_size=3).retrieve(
        "vitamin", k=2
    )

    assert [r.doc_id for r in out] == ["B", "A"]
    assert out[0].score == 5.0
    # Citation metadata from the base hit is preserved.
    assert out[0].chunk_id == "B#0"
    assert out[0].title == "title-B"


def test_pool_size_governs_base_overfetch_when_larger_than_k() -> None:
    base = FakeBase([_hit(str(i), 1.0 - i / 100) for i in range(60)])
    corpus = _corpus(*(Document(id=str(i), title="t", text="x") for i in range(60)))
    reranker = ScriptedReranker({})

    RerankingRetriever(base, reranker, corpus, pool_size=50).retrieve("q", k=10)

    assert base.calls == [("q", 50)]


def test_k_governs_overfetch_when_larger_than_pool() -> None:
    base = FakeBase([_hit(str(i), 1.0 - i / 100) for i in range(60)])
    corpus = _corpus(*(Document(id=str(i), title="t", text="x") for i in range(60)))
    reranker = ScriptedReranker({})

    RerankingRetriever(base, reranker, corpus, pool_size=20).retrieve("q", k=40)

    assert base.calls == [("q", 40)]


def test_empty_pool_short_circuits_reranker() -> None:
    base = FakeBase([])
    reranker = ScriptedReranker({"anything": 1.0})
    out = RerankingRetriever(base, reranker, _corpus(), pool_size=10).retrieve("q", k=5)
    assert out == []
    assert reranker.calls == []


def test_passage_falls_back_to_title_when_doc_missing() -> None:
    base = FakeBase([_hit("MISSING", 0.5)])
    reranker = ScriptedReranker({"title-MISSING": 7.0})
    out = RerankingRetriever(base, reranker, _corpus(), pool_size=10).retrieve("q", k=1)
    assert [r.doc_id for r in out] == ["MISSING"]
    assert out[0].score == 7.0


def test_default_pool_size_is_exposed_as_constant() -> None:
    base = FakeBase([_hit("A", 0.5)])
    RerankingRetriever(base, ScriptedReranker({}), _corpus()).retrieve("q", k=1)
    assert base.calls == [("q", DEFAULT_POOL_SIZE)]


def test_invalid_pool_size_rejected() -> None:
    with pytest.raises(ValueError, match="pool_size"):
        RerankingRetriever(FakeBase([]), ScriptedReranker({}), _corpus(), pool_size=0)
