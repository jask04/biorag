"""``ask`` CLI — single-shot grounded biomedical Q&A from the indexed corpus.

This is the minimal usable end-to-end surface for biorag: question in,
cited answer out. A richer interactive front-end with config toggles
lands on Day 11 as a Streamlit app; this module is the headless version
that exercises the same pipeline.

Usage:

    uv run ask "Does vitamin C reduce the duration of the common cold?"
    uv run ask -k 8 --hyde "..."
    uv run ask --retriever hybrid "..."
    uv run ask -v "..."           # also print every retrieved doc
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping

from biorag.bm25 import BM25Retriever
from biorag.corpus import Document, load_documents
from biorag.embed import CachedEmbedder, general_embedder
from biorag.generate import DEFAULT_K, Answer, GeminiAnswerGenerator
from biorag.hybrid import HybridRetriever
from biorag.rerank import RerankingRetriever, general_reranker
from biorag.retrieve import DenseRetriever, Retriever
from biorag.rewrite import GeminiHyDERewriter, HydeRetriever

RETRIEVERS = ("dense", "bm25", "hybrid", "hybrid+rerank")


def build_retriever(
    name: str,
    documents: list[Document],
    corpus: Mapping[str, Document],
    *,
    use_hyde: bool,
) -> Retriever:
    """Construct the retrieval pipeline named by ``--retriever`` (+ HyDE)."""
    bm25 = BM25Retriever(documents)
    if name == "dense":
        retriever: Retriever = DenseRetriever(CachedEmbedder(general_embedder()))
    elif name == "bm25":
        retriever = bm25
    elif name == "hybrid":
        retriever = HybridRetriever(
            DenseRetriever(CachedEmbedder(general_embedder())),
            bm25,
        )
    elif name == "hybrid+rerank":
        base = HybridRetriever(
            DenseRetriever(CachedEmbedder(general_embedder())),
            bm25,
        )
        retriever = RerankingRetriever(base, general_reranker(), corpus)
    else:
        raise ValueError(f"unknown retriever: {name}")

    if use_hyde:
        retriever = HydeRetriever(retriever, GeminiHyDERewriter())
    return retriever


def render_answer(answer: Answer, verbose: bool) -> str:
    """Format an :class:`Answer` for the terminal."""
    out: list[str] = [answer.text, ""]
    if answer.citations:
        out.append("Sources:")
        for citation in answer.citations:
            title = citation.title or "(untitled)"
            out.append(f"  [{citation.doc_id}] {title}")
    elif not answer.unsupported:
        out.append("(no citations parsed — answer may be unsupported)")
    if verbose:
        out.extend(["", "Retrieved (in pipeline order):"])
        for hit in answer.retrieved:
            title = hit.title or "(untitled)"
            out.append(f"  {hit.score:>7.3f}  [{hit.doc_id}]  {title}")
    return "\n".join(out)


def ask() -> None:
    """Console-script entry point. See module docstring for examples."""
    parser = argparse.ArgumentParser(prog="ask", description=__doc__)
    parser.add_argument("question", help="Question to ask the indexed corpus.")
    parser.add_argument(
        "-k", "--k", type=int, default=DEFAULT_K,
        help=f"Number of passages to ground the answer in (default: {DEFAULT_K}).",
    )
    parser.add_argument(
        "--retriever",
        default="hybrid+rerank",
        choices=RETRIEVERS,
        help="Retrieval pipeline. Default 'hybrid+rerank' (best on the eval).",
    )
    parser.add_argument(
        "--hyde",
        action="store_true",
        help="Wrap the retriever in HyDE query rewriting (costs 1 Gemini call).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Also print every retrieved doc with its score.",
    )
    args = parser.parse_args()

    documents = load_documents()
    corpus = {doc.id: doc for doc in documents}
    retriever = build_retriever(
        args.retriever, documents, corpus, use_hyde=args.hyde
    )
    hits = retriever.retrieve(args.question, k=args.k)
    generator = GeminiAnswerGenerator(corpus)
    answer = generator.answer(args.question, hits)
    print(render_answer(answer, verbose=args.verbose))


if __name__ == "__main__":
    ask()
