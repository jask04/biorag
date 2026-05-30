"""Run the retrieval eval harness over each configured retriever.

Assumes the normalized corpus (``scripts/build_corpus.py``) and any dense
indexes (``scripts/build_index.py --embedder {general,biomedical}``) have
already been built. Writes ``eval_results/retrieval.json`` and prints a
table.

Examples
--------
    uv run python scripts/eval_retrieval.py                       # all four
    uv run python scripts/eval_retrieval.py --retrievers bm25,hybrid
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
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
from biorag.retrieve import DenseRetriever, Retriever

ALL_NAMES = ("dense-general", "dense-biomedical", "bm25", "hybrid")


def _build(
    name: str, documents: list[Document], bm25: BM25Retriever
) -> Retriever:
    if name == "dense-general":
        return DenseRetriever(CachedEmbedder(general_embedder()))
    if name == "dense-biomedical":
        return DenseRetriever(CachedEmbedder(biomedical_embedder()))
    if name == "bm25":
        return bm25
    if name == "hybrid":
        return HybridRetriever(
            DenseRetriever(CachedEmbedder(general_embedder())),
            bm25,
        )
    raise ValueError(f"unknown retriever: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrievers",
        default=",".join(ALL_NAMES),
        help=f"Comma-separated retrievers to evaluate. Choices: {ALL_NAMES}.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RESULTS_DIR / "retrieval.json"),
        help="Where to write the JSON results.",
    )
    args = parser.parse_args()

    names = [n.strip() for n in args.retrievers.split(",") if n.strip()]
    for n in names:
        if n not in ALL_NAMES:
            raise SystemExit(f"unknown retriever {n!r}; pick from {ALL_NAMES}")

    documents = load_documents()
    queries = load_queries()
    qrels = load_qrels()
    bm25 = BM25Retriever(documents)  # built once, reused by bm25 + hybrid
    print(f"Loaded {len(documents)} docs, {len(queries)} queries, {len(qrels)} qrels")

    results: list[RetrieverEvalResult] = []
    for name in names:
        print(f"\n→ Evaluating {name} ...")
        retriever = _build(name, documents, bm25)
        result = evaluate(retriever, queries, qrels, name=name)
        results.append(result)
        # Flush per-retriever LLM/embed caches if the retriever owns one.
        save: Callable[[], None] | None = getattr(
            getattr(retriever, "embedder", None), "save", None
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
