"""biorag — Streamlit front-end.

Day 11 ships the **Ask** tab: question box + a pipeline config panel
(retriever mode, rerank, HyDE, top-k) → a grounded answer with cited
sources. Day 12 adds the **Benchmark** tab over ``eval_results/``.

Heavy resources (corpus, BM25 index, embedder, cross-encoder) are built
once per process via ``st.cache_resource`` and shared across reruns —
a Streamlit rerun executes this whole file top to bottom, so anything
not cached would rebuild on every widget interaction.

Run locally:

    uv run streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from biorag.bm25 import BM25Retriever
from biorag.corpus import Document, load_documents
from biorag.embed import CachedEmbedder, general_embedder
from biorag.generate import Answer, GeminiAnswerGenerator
from biorag.hybrid import HybridRetriever
from biorag.rerank import RerankingRetriever, general_reranker
from biorag.retrieve import DenseRetriever, Retriever
from biorag.rewrite import GeminiHyDERewriter, HydeRetriever

st.set_page_config(
    page_title="biorag — cited biomedical Q&A",
    page_icon="🧬",
    layout="wide",
)


# ---------- cached resources (built once per process) ----------


@st.cache_resource(show_spinner="Loading corpus …")
def get_documents() -> list[Document]:
    return load_documents()


@st.cache_resource(show_spinner=False)
def get_corpus() -> dict[str, Document]:
    return {d.id: d for d in get_documents()}


@st.cache_resource(show_spinner="Building BM25 index …")
def get_bm25() -> BM25Retriever:
    return BM25Retriever(get_documents())


@st.cache_resource(show_spinner="Loading embedding model …")
def get_dense() -> DenseRetriever:
    return DenseRetriever(CachedEmbedder(general_embedder()))


@st.cache_resource(show_spinner="Loading cross-encoder …")
def get_reranker_model():  # type: ignore[no-untyped-def]
    return general_reranker()


@st.cache_resource(show_spinner=False)
def get_generator() -> GeminiAnswerGenerator:
    return GeminiAnswerGenerator(get_corpus())


@st.cache_resource(show_spinner=False)
def get_rewriter() -> GeminiHyDERewriter:
    return GeminiHyDERewriter()


def build_pipeline(mode: str, use_rerank: bool, use_hyde: bool) -> Retriever:
    """Assemble the retriever stack the sidebar describes."""
    retriever: Retriever
    if mode == "dense":
        retriever = get_dense()
    elif mode == "bm25":
        retriever = get_bm25()
    else:
        retriever = HybridRetriever(get_dense(), get_bm25())
    if use_rerank:
        retriever = RerankingRetriever(
            retriever, get_reranker_model(), get_corpus()
        )
    if use_hyde:
        retriever = HydeRetriever(retriever, get_rewriter())
    return retriever


# ---------- sidebar: pipeline config ----------

with st.sidebar:
    st.title("🧬 biorag")
    st.caption(
        "Cited Q&A over biomedical literature (BEIR NFCorpus). "
        "A research-literature assistant — **not** medical advice."
    )
    st.divider()
    st.subheader("Pipeline")
    mode = st.radio(
        "Retriever",
        options=("hybrid", "dense", "bm25"),
        help=(
            "dense = BGE-small embeddings via Qdrant · bm25 = lexical · "
            "hybrid = both fused with reciprocal rank fusion"
        ),
    )
    use_rerank = st.toggle(
        "Cross-encoder rerank",
        value=True,
        help="Re-score the candidate pool with MS-MARCO MiniLM. "
        "Best MRR/nDCG on the benchmark.",
    )
    use_hyde = st.toggle(
        "HyDE query rewriting",
        value=False,
        help="Rewrite the query into a hypothetical answer passage with "
        "Gemini before retrieving. Costs one extra LLM call.",
    )
    top_k = st.slider("Passages to ground on (k)", 3, 10, 5)
    st.divider()
    st.caption(
        "Pipeline numbers live in the eval harness — see the README "
        "results table."
    )


# ---------- main: ask tab ----------

st.header("Ask the biomedical literature")

question = st.text_input(
    "Question",
    placeholder="e.g. What are the cardiovascular benefits of the Mediterranean diet?",
    label_visibility="collapsed",
)

ask_clicked = st.button("Ask", type="primary", disabled=not question.strip())

if ask_clicked and question.strip():
    pipeline = build_pipeline(mode, use_rerank, use_hyde)
    with st.spinner("Retrieving …"):
        hits = pipeline.retrieve(question.strip(), k=top_k)
    with st.spinner("Generating grounded answer …"):
        answer: Answer = get_generator().answer(question.strip(), hits)

    if answer.unsupported:
        st.warning(
            "The retrieved passages don't address this question, so no "
            "answer was generated. Try rephrasing, or toggle the pipeline "
            "options — the corpus (NFCorpus) is nutrition-focused and "
            "doesn't cover every biomedical topic."
        )
    else:
        st.markdown(answer.text)

    if answer.citations:
        st.subheader("Sources")
        for citation in answer.citations:
            doc = get_corpus().get(citation.doc_id)
            with st.expander(f"[{citation.doc_id}] {citation.title or '(untitled)'}"):
                st.write(doc.text if doc else "(document text unavailable)")

    with st.expander(f"Retrieved passages ({len(hits)})", expanded=False):
        for hit in hits:
            cited = any(c.doc_id == hit.doc_id for c in answer.citations)
            marker = "✓ cited" if cited else "not cited"
            st.markdown(
                f"**[{hit.doc_id}]** {hit.title or '(untitled)'}  \n"
                f"score `{hit.score:.3f}` · {marker}"
            )
else:
    st.info(
        "Ask a question about nutrition / biomedical research. Answers are "
        "generated **only** from retrieved NFCorpus abstracts and cite "
        "their sources inline."
    )
