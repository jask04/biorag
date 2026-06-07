"""Run the answer-quality eval over one or more pipeline configurations.

Designed to be re-runnable under a tight LLM quota: every Gemini call
goes through a LangChain SQLite cache, so today's quota only pays for
work that wasn't done before. Re-run tomorrow with a larger ``--n`` and
only the new questions will hit the API.

Examples
--------
    uv run python scripts/eval_answers.py
    uv run python scripts/eval_answers.py --n 5 --config hybrid+rerank
    uv run python scripts/eval_answers.py --include-context-precision
"""

from __future__ import annotations

import argparse
from pathlib import Path

from biorag.bm25 import BM25Retriever
from biorag.corpus import load_documents, load_qrels, load_queries
from biorag.embed import CachedEmbedder, general_embedder
from biorag.eval.answers import (
    DEFAULT_K,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SAMPLE_SIZE,
    AnswerEvalResult,
    evaluate_pipeline,
    install_langchain_sqlite_cache,
    save_results,
    select_sample,
)
from biorag.generate import GeminiAnswerGenerator
from biorag.hybrid import HybridRetriever
from biorag.rerank import RerankingRetriever, general_reranker
from biorag.retrieve import DenseRetriever, Retriever

CONFIGS = ("hybrid", "hybrid+rerank")


def build_pipeline(name: str, docs: list, corpus: dict) -> Retriever:  # type: ignore[type-arg]
    bm25 = BM25Retriever(docs)
    hybrid = HybridRetriever(
        DenseRetriever(CachedEmbedder(general_embedder())),
        bm25,
    )
    if name == "hybrid":
        return hybrid
    if name == "hybrid+rerank":
        return RerankingRetriever(hybrid, general_reranker(), corpus)
    raise ValueError(f"unknown config: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Sample size (deterministic prefix of qrel-positive queries). "
             f"Default: {DEFAULT_SAMPLE_SIZE}.",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"Top-k passages per question. Default: {DEFAULT_K}.",
    )
    parser.add_argument(
        "--configs", default=",".join(CONFIGS),
        help=f"Comma-separated configs to evaluate. Choices: {CONFIGS}.",
    )
    parser.add_argument(
        "--include-context-precision", action="store_true",
        help="Add ragas context_precision (costs k extra LLM calls per question).",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_RESULTS_DIR / "answers.json"),
    )
    args = parser.parse_args()

    install_langchain_sqlite_cache()

    documents = load_documents()
    queries = load_queries()
    qrels = load_qrels()
    corpus = {d.id: d for d in documents}
    sample = select_sample(queries, qrels, n=args.n)
    print(
        f"Loaded {len(documents)} docs / {len(queries)} queries / "
        f"{len(qrels)} qrels — sampling first {len(sample)} qrel-positive queries"
    )

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    for name in config_names:
        if name not in CONFIGS:
            raise SystemExit(f"unknown config {name!r}; pick from {CONFIGS}")

    generator = GeminiAnswerGenerator(corpus)
    results: list[AnswerEvalResult] = []
    for name in config_names:
        print(f"\n→ Evaluating {name} (k={args.k}) ...")
        retriever = build_pipeline(name, documents, corpus)
        result = evaluate_pipeline(
            name,
            sample,
            retriever,
            generator,
            corpus,
            k=args.k,
            include_context_precision=args.include_context_precision,
        )
        results.append(result)
        scored = sum(1 for r in result.per_question if r.get("scores"))
        print(
            f"   answered {result.sample_size}/{len(sample)} | "
            f"scored {scored} | metrics: {result.metrics or '(none)'}"
        )
        for note in result.notes:
            print(f"   note: {note}")

    out = save_results(results, path=Path(args.output))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
