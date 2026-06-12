"""End-to-end pipeline integration test on a tiny in-memory fixture corpus.

This is the one test that wires the real components together —
chunking, BM25, hybrid fusion (RRF), cross-encoder reranking, and the
citation-grounding helpers — and runs a question all the way to a cited
answer. It deliberately uses **no network and no model downloads** so it
runs in CI:

* BM25 is genuinely in-memory.
* The dense channel is a small deterministic fake embedder (so hybrid
  fusion and the ``Retriever`` protocol are exercised without Qdrant).
* The reranker is a deterministic keyword-overlap fake.
* The LLM is mocked by a stub generator that reuses the *real*
  ``format_passages`` / ``extract_citations`` / ``build_citations``
  helpers from :mod:`biorag.generate`, so the citation contract is
  tested for real — only the Gemini network call is replaced.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from biorag.bm25 import BM25Retriever
from biorag.chunk import chunk_documents
from biorag.corpus import Document
from biorag.embed import EmbeddingMatrix
from biorag.generate import (
    Answer,
    build_citations,
    extract_citations,
    format_passages,
)
from biorag.hybrid import HybridRetriever
from biorag.rerank import RerankingRetriever
from biorag.retrieve import RetrievalResult, Retriever

FIXTURE_DOCS = [
    Document(
        id="MED-1",
        title="Vitamin C and the common cold",
        text="Randomized trials of ascorbic acid supplementation report a "
        "modest reduction in the duration of common cold symptoms.",
    ),
    Document(
        id="MED-2",
        title="Mediterranean diet and cardiovascular outcomes",
        text="Adherence to a Mediterranean diet is associated with lower "
        "incidence of cardiovascular disease and stroke.",
    ),
    Document(
        id="MED-3",
        title="Curcumin bioavailability",
        text="Oral curcumin has poor systemic bioavailability; piperine "
        "co-administration increases plasma concentrations.",
    ),
    Document(
        id="MED-4",
        title="Vitamin D and respiratory infection",
        text="Vitamin D supplementation may reduce the risk of acute "
        "respiratory tract infections in deficient individuals.",
    ),
    Document(
        id="MED-5",
        title="Dietary fiber and colorectal cancer",
        text="Higher dietary fiber intake is associated with a reduced risk "
        "of colorectal cancer in cohort studies.",
    ),
]


class KeywordEmbedder:
    """Deterministic bag-of-words embedder over a fixed vocabulary.

    Real enough to make cosine similarity meaningful for the fixture —
    documents sharing query terms get nearer vectors — without any model
    download. Satisfies the ``Embedder`` protocol.
    """

    def __init__(self, vocab: Sequence[str]) -> None:
        self.name = "keyword-embedder"
        self._vocab = list(vocab)
        self.dimension = len(self._vocab)

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            lowered = text.lower()
            for j, term in enumerate(self._vocab):
                out[i, j] = float(lowered.count(term))
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


class InMemoryDenseRetriever:
    """Dense retrieval over the fixture, scored by cosine, no Qdrant."""

    def __init__(self, docs: Sequence[Document], embedder: KeywordEmbedder) -> None:
        self._docs = list(docs)
        self._embedder = embedder
        chunks = list(chunk_documents(self._docs))
        self._chunks = chunks
        self._matrix = embedder.embed([c.text for c in chunks])

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        qvec = self._embedder.embed([query])[0]
        scores = self._matrix @ qvec
        order = np.argsort(-scores)
        results: list[RetrievalResult] = []
        seen: set[str] = set()
        for idx in order:
            chunk = self._chunks[idx]
            if chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            results.append(
                RetrievalResult(
                    doc_id=chunk.doc_id,
                    score=float(scores[idx]),
                    chunk_id=chunk.id,
                    title=next(d.title for d in self._docs if d.id == chunk.doc_id),
                )
            )
            if len(results) >= k:
                break
        return results


class KeywordOverlapReranker:
    """Scores passages by shared-token overlap with the query."""

    name = "keyword-overlap-reranker"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q_terms = set(query.lower().split())
        return [
            float(len(q_terms & set(p.lower().split()))) for p in passages
        ]


class StubGenerator:
    """Mock LLM that reuses the real citation helpers.

    Mimics a well-behaved Gemini response: it 'answers' by citing the
    top retrieved doc, threading the id through the real
    ``extract_citations`` / ``build_citations`` path so the grounding
    contract is exercised for real.
    """

    name = "stub-generator"

    def __init__(self, corpus: dict[str, Document]) -> None:
        self._corpus = corpus

    def answer(self, query: str, hits: Sequence[RetrievalResult]) -> Answer:
        if not hits:
            return Answer(query=query, text="No passages.", retrieved=list(hits))
        # Render the passages the way the real generator would, then
        # produce canned prose that cites the top hit (plus a deliberately
        # invalid id that the parser must drop).
        _ = format_passages(hits, self._corpus)
        top = hits[0]
        text = f"Based on the literature, {top.title.lower()} is relevant "
        text += f"[{top.doc_id}]. An unsupported claim [MED-9999]."
        cited = extract_citations(text, [h.doc_id for h in hits])
        citations = build_citations(cited, self._corpus, hits)
        return Answer(
            query=query, text=text, citations=citations, retrieved=list(hits)
        )


def _build_pipeline() -> tuple[Retriever, StubGenerator, dict[str, Document]]:
    corpus = {d.id: d for d in FIXTURE_DOCS}
    vocab = [
        "vitamin", "ascorbic", "cold", "mediterranean", "diet",
        "cardiovascular", "curcumin", "respiratory", "fiber", "cancer",
    ]
    dense = InMemoryDenseRetriever(FIXTURE_DOCS, KeywordEmbedder(vocab))
    bm25 = BM25Retriever(FIXTURE_DOCS)
    hybrid = HybridRetriever(dense, bm25)
    reranked = RerankingRetriever(hybrid, KeywordOverlapReranker(), corpus)
    return reranked, StubGenerator(corpus), corpus


def test_end_to_end_retrieves_and_cites_the_right_document() -> None:
    pipeline, generator, _ = _build_pipeline()

    hits = pipeline.retrieve("vitamin C common cold ascorbic", k=3)
    assert hits, "pipeline returned no hits"
    assert hits[0].doc_id == "MED-1"  # the vitamin-C/cold doc tops the ranking

    answer = generator.answer("vitamin C common cold ascorbic", hits)
    assert not answer.unsupported
    # The real citation parser kept the valid id and dropped the fake one.
    assert [c.doc_id for c in answer.citations] == ["MED-1"]
    assert "MED-9999" not in [c.doc_id for c in answer.citations]
    assert answer.citations[0].title == "Vitamin C and the common cold"


def test_end_to_end_routes_a_different_topic_to_its_document() -> None:
    pipeline, generator, _ = _build_pipeline()

    hits = pipeline.retrieve("mediterranean diet cardiovascular", k=3)
    assert hits[0].doc_id == "MED-2"

    answer = generator.answer("mediterranean diet cardiovascular", hits)
    assert answer.citations[0].doc_id == "MED-2"


def test_end_to_end_hybrid_beats_neither_channel_empty() -> None:
    # Sanity: every stage in the stack returns the protocol type and a
    # non-empty ranking for an in-vocabulary query.
    pipeline, _, _ = _build_pipeline()
    assert isinstance(pipeline, Retriever)
    hits = pipeline.retrieve("dietary fiber colorectal cancer", k=5)
    assert hits[0].doc_id == "MED-5"
    assert all(isinstance(h, RetrievalResult) for h in hits)
