"""Run the retrieval eval harness over each configured retriever.

Assumes the normalized corpus (``scripts/build_corpus.py``) and any dense
indexes (``scripts/build_index.py --embedder {general,biomedical}``) have
already been built. Writes ``eval_results/retrieval.json`` and prints a
table.

Configurations include both raw retrievers and reranked variants — the
rerank lift is one of the headline findings the harness is built to
quantify.

Examples
--------
    uv run python scripts/eval_retrieval.py                       # all
    uv run python scripts/eval_retrieval.py --retrievers bm25,hybrid
    uv run python scripts/eval_retrieval.py --retrievers hybrid,hybrid+rerank-mini
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path

from biorag.bm25 import BM25Retriever
from biorag.corpus import Document, load_documents, load_qrels, load_queries
from biorag.embed import (
    CachedEmbedder,
    biomedical_embedder,
    general_embedder,
)
from biorag.eval.retrieval import (
    DEFAULT_RESULTS_DIR,
    RetrieverEvalResult,
    evaluate,
    render_table,
    save_results,
)
from biorag.hybrid import HybridRetriever
from biorag.rerank import (
    CrossEncoderReranker,
    Reranker,
    RerankingRetriever,
    biomedical_reranker,
    general_reranker,
)
from biorag.retrieve import DenseRetriever, Retriever
from biorag.rewrite import GeminiHyDERewriter, HydeRetriever, Rewriter

ALL_NAMES = (
    "dense-general",
    "dense-biomedical",
    "bm25",
    "hybrid",
    "dense-general+rerank-mini",
    "hybrid+rerank-mini",
    "hybrid+rerank-medcpt",
    "hybrid+hyde",
    "hybrid+hyde+rerank-mini",
)

# Configurations evaluated by default. MedCPT is the heaviest cross-encoder
# in the project (BERT-base, CPU-only because of the MPS sync bug) — it's
# kept as an opt-in via ``--retrievers`` rather than the default sweep.
DEFAULT_NAMES = tuple(n for n in ALL_NAMES if n != "hybrid+rerank-medcpt")


class _RerankerCache:
    """Build each cross-encoder at most once per run (they're heavy)."""

    def __init__(self) -> None:
        self._mini: CrossEncoderReranker | None = None
        self._medcpt: CrossEncoderReranker | None = None

    def mini(self) -> Reranker:
        if self._mini is None:
            self._mini = general_reranker()
        return self._mini

    def medcpt(self) -> Reranker:
        if self._medcpt is None:
            self._medcpt = biomedical_reranker()
        return self._medcpt


class _RewriterCache:
    """Build the HyDE rewriter once and reuse it across HyDE configs.

    All HyDE configs share the same rewrite cache on disk anyway, but
    sharing the instance means the in-memory cache is hot for the second
    pass and we never pay for the SDK ``Client(api_key=...)`` twice.
    """

    def __init__(self) -> None:
        self._hyde: GeminiHyDERewriter | None = None

    def hyde(self) -> Rewriter:
        if self._hyde is None:
            self._hyde = GeminiHyDERewriter()
        return self._hyde


def _hybrid(bm25: BM25Retriever) -> HybridRetriever:
    return HybridRetriever(
        DenseRetriever(CachedEmbedder(general_embedder())),
        bm25,
    )


def _build(
    name: str,
    documents: list[Document],
    corpus: Mapping[str, Document],
    bm25: BM25Retriever,
    rerankers: _RerankerCache,
    rewriters: _RewriterCache,
) -> Retriever:
    if name == "dense-general":
        return DenseRetriever(CachedEmbedder(general_embedder()))
    if name == "dense-biomedical":
        return DenseRetriever(CachedEmbedder(biomedical_embedder()))
    if name == "bm25":
        return bm25
    if name == "hybrid":
        return _hybrid(bm25)
    if name == "dense-general+rerank-mini":
        base = DenseRetriever(CachedEmbedder(general_embedder()))
        return RerankingRetriever(base, rerankers.mini(), corpus)
    if name == "hybrid+rerank-mini":
        return RerankingRetriever(_hybrid(bm25), rerankers.mini(), corpus)
    if name == "hybrid+rerank-medcpt":
        return RerankingRetriever(_hybrid(bm25), rerankers.medcpt(), corpus)
    if name == "hybrid+hyde":
        return HydeRetriever(_hybrid(bm25), rewriters.hyde())
    if name == "hybrid+hyde+rerank-mini":
        hyde = HydeRetriever(_hybrid(bm25), rewriters.hyde())
        return RerankingRetriever(hyde, rerankers.mini(), corpus)
    raise ValueError(f"unknown retriever: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrievers",
        default=",".join(DEFAULT_NAMES),
        help=f"Comma-separated retrievers to evaluate. Choices: {ALL_NAMES}.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RESULTS_DIR / "retrieval.json"),
        help="Where to write the JSON results.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help=(
            "Cap the number of scored queries per retriever (useful when an "
            "LLM-backed config has to fit inside a daily quota)."
        ),
    )
    args = parser.parse_args()

    names = [n.strip() for n in args.retrievers.split(",") if n.strip()]
    for n in names:
        if n not in ALL_NAMES:
            raise SystemExit(f"unknown retriever {n!r}; pick from {ALL_NAMES}")

    documents = load_documents()
    queries = load_queries()
    qrels = load_qrels()
    corpus = {d.id: d for d in documents}
    bm25 = BM25Retriever(documents)  # built once, reused by bm25 + hybrid
    rerankers = _RerankerCache()
    rewriters = _RewriterCache()
    print(f"Loaded {len(documents)} docs, {len(queries)} queries, {len(qrels)} qrels")

    results: list[RetrieverEvalResult] = []
    for name in names:
        print(f"\n→ Evaluating {name} ...")
        retriever = _build(name, documents, corpus, bm25, rerankers, rewriters)
        result = evaluate(
            retriever, queries, qrels, name=name, max_queries=args.max_queries
        )
        results.append(result)
        # Flush per-retriever embed caches if the retriever (or its base)
        # owns one — RerankingRetriever stores the base privately.
        for candidate in (retriever, getattr(retriever, "_base", None)):
            save: Callable[[], None] | None = getattr(
                getattr(candidate, "embedder", None), "save", None
            )
            if callable(save):
                save()
        print(f"   {result.queries} queries scored")

    print()
    print(render_table(results))
    out = save_results(results, path=Path(args.output))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
