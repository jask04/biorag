"""Cross-encoder reranking as an optional pipeline stage.

Bi-encoders (the dense embedders behind ``DenseRetriever``) encode the
query and a document independently — fast and cacheable, but they only
see one side at a time. Cross-encoders score ``(query, passage)`` pairs
jointly, which trades throughput for precision. The standard recipe is
to use a bi-encoder for the initial ``top-N`` recall and a cross-encoder
to re-rank that pool down to a smaller ``top-k``.

This module exposes a :class:`Reranker` protocol, a
:class:`CrossEncoderReranker` adapter for ``sentence-transformers``
cross-encoders, and :class:`RerankingRetriever` — a retriever wrapper
that composes any base :class:`~biorag.retrieve.Retriever` with a
reranker while preserving the ``Retriever`` shape, so the eval harness
and downstream callers treat ``hybrid+rerank`` the same as ``hybrid``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, Protocol, runtime_checkable

from biorag.corpus import Document
from biorag.retrieve import RetrievalResult, Retriever

GENERAL_RERANKER_MODEL: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BIOMEDICAL_RERANKER_MODEL: Final[str] = "ncbi/MedCPT-Cross-Encoder"
DEFAULT_POOL_SIZE: Final[int] = 30


@runtime_checkable
class Reranker(Protocol):
    """Scores ``(query, passage)`` pairs jointly. Larger is more relevant."""

    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Adapter around any ``sentence-transformers`` cross-encoder model."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        # Lazy imports so importing ``biorag.rerank`` is cheap and tests
        # that use fakes don't pay for torch / cross-encoder weights.
        import torch
        from sentence_transformers import CrossEncoder

        if device is None:
            # CPU is the safe default for cross-encoders: torch's MPS
            # backend has a Metal-stream synchronization hang on the
            # tensor copies inside CrossEncoder.predict on this
            # macOS/torch combo (verified empirically). The bi-encoder
            # path is fine on MPS, but rerank is — pick CPU here and
            # let callers opt into ``device='mps'`` explicitly.
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = model_name
        self.device = device
        self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        raw = self._model.predict(pairs, show_progress_bar=False)
        return [float(x) for x in raw]


def general_reranker() -> CrossEncoderReranker:
    """MS-MARCO MiniLM — small, fast, the standard general-domain baseline."""
    return CrossEncoderReranker(GENERAL_RERANKER_MODEL)


def biomedical_reranker() -> CrossEncoderReranker:
    """MedCPT — NCBI's biomedical cross-encoder; bigger but domain-tuned."""
    return CrossEncoderReranker(BIOMEDICAL_RERANKER_MODEL)


class RerankingRetriever:
    """Compose a base retriever with a reranker, preserving the protocol.

    ``retrieve(query, k)`` pulls ``max(pool_size, k)`` candidates from the
    base retriever, asks the reranker to score the ``(query, doc text)``
    pairs, and returns the top-k after re-sorting by reranker score.
    Document text comes from ``corpus`` (a ``doc_id`` → :class:`Document`
    map) so passages match what callers see at citation time — not the
    chunk slice, which would let the score depend on the chunker's
    settings.
    """

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        corpus: Mapping[str, Document],
        pool_size: int = DEFAULT_POOL_SIZE,
    ) -> None:
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        self._base = base
        self._reranker = reranker
        self._corpus = corpus
        self._pool_size = pool_size

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        pool = self._base.retrieve(query, k=max(self._pool_size, k))
        if not pool:
            return []
        passages = [_passage_for(self._corpus, hit) for hit in pool]
        scores = self._reranker.score(query, passages)
        reordered = sorted(zip(pool, scores, strict=True), key=lambda x: -x[1])
        return [
            RetrievalResult(
                doc_id=hit.doc_id,
                score=float(score),
                chunk_id=hit.chunk_id,
                title=hit.title,
            )
            for hit, score in reordered[:k]
        ]


def _passage_for(corpus: Mapping[str, Document], hit: RetrievalResult) -> str:
    """Document text the reranker sees. Falls back to title if doc missing."""
    doc = corpus.get(hit.doc_id)
    if doc is None:
        return hit.title
    title = doc.title.strip()
    text = doc.text.strip()
    if title and text:
        return f"{title}. {text}"
    return title or text


__all__ = [
    "BIOMEDICAL_RERANKER_MODEL",
    "DEFAULT_POOL_SIZE",
    "GENERAL_RERANKER_MODEL",
    "CrossEncoderReranker",
    "Reranker",
    "RerankingRetriever",
    "biomedical_reranker",
    "general_reranker",
]
