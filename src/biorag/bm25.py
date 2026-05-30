"""BM25 lexical retrieval over the document corpus.

A deliberately simple, in-memory ``rank-bm25`` index. NFCorpus is small
(~3.6k docs) so an explicit Python BM25 fits comfortably in memory and
rebuilds in well under a second — no need for a server.

Tokenization is intentionally minimal (lowercase + ``\\w+`` extraction),
matching what a fresh reader expects from a "BM25 baseline". The point of
this module isn't to win on lexical search alone; it's to be the
controllable lexical channel in the hybrid + RRF fusion (see
:mod:`biorag.hybrid`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from rank_bm25 import BM25Okapi

from biorag.corpus import Document
from biorag.retrieve import RetrievalResult

DEFAULT_K1: Final[float] = 1.5
DEFAULT_B: Final[float] = 0.75

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase, word-boundary tokenization shared by index and query.

    Kept as a module-level function so callers can reuse the exact same
    rule when constructing custom corpora or debugging mismatches.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """In-memory BM25 retriever over a fixed document corpus."""

    def __init__(
        self,
        documents: Sequence[Document],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        if not documents:
            raise ValueError("BM25Retriever requires at least one document")
        self._documents = list(documents)
        self._tokenized = [
            tokenize(f"{doc.title} {doc.text}") for doc in self._documents
        ]
        # Empty token lists confuse BM25Okapi's IDF math; substitute a
        # single sentinel so the doc still occupies an index but never
        # scores above zero on real queries.
        self._tokenized = [tokens or ["\x00"] for tokens in self._tokenized]
        self._bm25 = BM25Okapi(self._tokenized, k1=k1, b=b)

    def __len__(self) -> int:
        return len(self._documents)

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        # Argsort descending; stable on score ties via doc-order index.
        order = sorted(
            range(len(scores)),
            key=lambda i: (-float(scores[i]), i),
        )
        results: list[RetrievalResult] = []
        for idx in order[:k]:
            score = float(scores[idx])
            if score <= 0.0:
                # BM25 scores can be zero/negative when no query token hits
                # the doc — these are not "retrieved" in any useful sense.
                break
            doc = self._documents[idx]
            results.append(
                RetrievalResult(
                    doc_id=doc.id,
                    score=score,
                    chunk_id=f"{doc.id}#0",
                    title=doc.title,
                )
            )
        return results


__all__ = ["DEFAULT_B", "DEFAULT_K1", "BM25Retriever", "tokenize"]
