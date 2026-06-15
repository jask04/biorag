# biorag

[![CI](https://github.com/jask04/biorag/actions/workflows/ci.yml/badge.svg)](https://github.com/jask04/biorag/actions/workflows/ci.yml)

Evaluation-first biomedical RAG: ask natural-language questions over
biomedical literature and get cited, grounded answers — backed by a
configurable retrieval pipeline and a reproducible benchmark harness that
measures every technique against a standard IR dataset.

biorag is a **cited literature Q&A assistant, not a medical-advice tool.**
Every answer is generated only from retrieved documents and cites them
inline.

**Live demo:** **https://huggingface.co/spaces/jask04/biorag** (Hugging
Face Space)

## The headline: every technique, measured

The point of biorag isn't that it does retrieval — it's that it *measures*
whether each technique earns its place. All numbers below are produced by
`scripts/eval_retrieval.py` against **BEIR NFCorpus** gold relevance
judgments (3,633 documents, 323 test queries with positive qrels).

### Retrieval quality (323 NFCorpus test queries)

| Configuration | Recall@10 | Recall@20 | MRR | nDCG@10 |
|---|---|---|---|---|
| BM25 (lexical) | 0.1522 | 0.1718 | 0.5103 | 0.3064 |
| Dense — biomedical (S-PubMedBert) | 0.1426 | 0.1765 | 0.5344 | 0.3161 |
| Dense — general (BGE-small) | 0.1609 | 0.2003 | 0.5392 | 0.3405 |
| **Hybrid** (dense + BM25, RRF) | **0.1688** | **0.2047** | 0.5700 | 0.3536 |
| Dense + cross-encoder rerank | 0.1683 | 0.1989 | **0.5813** | **0.3577** |
| Hybrid + cross-encoder rerank | 0.1680 | 0.2007 | 0.5795 | **0.3577** |

**What the numbers say:**

- **Hybrid fusion earns its place on recall.** Reciprocal Rank Fusion of
  the dense and lexical channels beats either one alone on Recall@10/@20 —
  the channels surface different relevant documents.
- **Cross-encoder reranking earns its place on ranking quality.** It lifts
  MRR and nDCG@10 (better *ordering* of the top results) while leaving
  recall roughly flat — exactly as expected, since it reorders an existing
  candidate pool rather than recalling new documents.
- **The general embedder beats the biomedical one here.** A slightly
  counter-intuitive but honest result: `BGE-small` (general, but heavily
  retrieval-tuned) outperforms `S-PubMedBert` (domain-pretrained) on every
  metric. Domain pretraining didn't outweigh retrieval-specific training
  on this dataset.

### Query rewriting (HyDE) — 21-query subset

HyDE rewrites each query into a hypothetical answer passage with an LLM
before retrieving, so these configurations cost one Gemini call per query.
To stay within the free-tier daily quota they're benchmarked on a fixed
**21-query subset** (same subset for every row — apples-to-apples, but
treat absolute values as directional, not definitive).

| Configuration | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| Hybrid | 0.0716 | 0.7122 | 0.3529 |
| Hybrid + rerank | 0.0759 | 0.7090 | 0.3979 |
| Hybrid + HyDE | 0.0668 | 0.6668 | 0.3254 |
| **Hybrid + HyDE + rerank** | 0.0746 | **0.8367** | **0.4216** |

On this subset, HyDE *alone* slightly hurts hybrid (the hypothetical
passage drifts off-topic for queries that lexical match already nails),
but **HyDE + reranking is the strongest configuration on MRR and
nDCG@10** — the cross-encoder filters HyDE's noisier candidate pool
effectively.

### Answer quality (ragas)

End-to-end answer scoring with [ragas](https://github.com/explodinggradients/ragas)
(LLM-as-judge) for **faithfulness** (are the answer's claims supported by
the retrieved context) and **answer relevancy** (does the answer address
the question). This calls the LLM several times per question, so on the
free tier it runs on a very small sample — see
[Honest limitations](#honest-limitations). The harness, caching, and
sampling are built to scale to the full set the moment more quota is
available (`scripts/eval_answers.py --n 50`).

## What it does

```
                       ┌─────────────────────────────────────────┐
  INGEST / INDEX       │  NFCorpus → chunk → embed → Qdrant       │
                       │                  └→ BM25 (in-memory)     │
                       └─────────────────────────────────────────┘
                                          │
  QUERY        question ─┐                ▼
               ┌─────────┴──────────────────────────────────────┐
               │  [HyDE rewrite?] → dense ⊕ BM25 (RRF) →         │
               │  [cross-encoder rerank?] → top-k passages       │
               └────────────────────────┬───────────────────────┘
                                         ▼
                          Gemini (grounded, cited) → answer [doc_id]
                                         │
  EVAL         ┌──────────────────────────┴──────────────────────┐
               │ retrieval: Recall@k / MRR / nDCG@10 vs gold qrels │
               │ answers:   ragas faithfulness / answer relevancy  │
               └─────────────────────────────────────────────────┘
```

Two surfaces (`app.py`, Streamlit):

- **Ask** — a question box plus a live pipeline-config panel (retriever
  mode, rerank on/off, HyDE on/off, top-k). Returns a grounded answer with
  inline `[doc_id]` citations; cited and retrieved passages are shown so
  you can see the grounding. Bring your own Gemini key, or use the shared
  demo key until its daily quota runs out.
- **Benchmark** — the results tables above, rendered live from
  `eval_results/` with the best value per metric highlighted.

## Design notes

- **Hybrid + RRF, by hand.** Reciprocal Rank Fusion is implemented
  directly (`src/biorag/hybrid.py`) rather than pulled from a library — it
  combines *ranks*, not scores, so it sidesteps the cosine-vs-BM25 scale
  mismatch. Tested against closed-form expected values.
- **Two-stage retrieval.** A fast bi-encoder recalls a candidate pool; a
  slower cross-encoder re-scores `(query, passage)` jointly. The eval
  harness quantifies the lift so the extra cost is justified by numbers.
- **Grounding is a hard contract.** The generation prompt requires a
  citation per claim and a fixed "unsupported" sentinel when the passages
  don't answer the question; the parser validates cited ids against the
  retrieved set, so a hallucinated id never reaches the structured output.
- **Free-tier discipline.** Retrieval eval is fully local (embeddings +
  gold qrels), runs over the entire query set, and caches embeddings.
  LLM-dependent steps (HyDE, answer eval) sample a fixed subset and cache
  every call to disk, so re-runs are free and resumable.

## Honest limitations

- **Single dataset.** Everything is measured on NFCorpus (nutrition-
  focused biomedical IR). Cross-dataset generalization (SciFact,
  TREC-COVID) is future work.
- **Sampled, small-N answer eval.** ragas and HyDE numbers come from small
  samples because the Gemini free tier caps requests at 20/day on this
  project. They're directional. The harness is built to scale to 50–100
  questions the moment quota allows (`--n`).
- **The demo's shared key is rate-limited.** Paste your own free Gemini key
  in the sidebar for unlimited use.

## Stack

| Layer | Choice |
|---|---|
| Language / tooling | Python 3.12, [uv](https://github.com/astral-sh/uv), ruff, mypy (strict) |
| Embeddings | sentence-transformers — `BAAI/bge-small-en-v1.5` (general) vs `pritamdeka/S-PubMedBert-MS-MARCO` (biomedical) |
| Lexical / fusion | `rank-bm25` + hand-written Reciprocal Rank Fusion |
| Vector store | Qdrant Cloud |
| Reranker | sentence-transformers CrossEncoder (`ms-marco-MiniLM`) |
| Query rewriting | HyDE via Gemini |
| Generation | Google `gemini-2.5-flash-lite` (`google-genai`) |
| Eval | custom retrieval harness + [ragas](https://github.com/explodinggradients/ragas) |
| Data | BEIR NFCorpus (corpus + queries + gold qrels) |
| UI / deploy | Streamlit on Hugging Face Spaces |
| CI | GitHub Actions (ruff + mypy + pytest) |

## Running locally

```bash
uv sync
cp .env.example .env          # add GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY

uv run python scripts/build_corpus.py            # download NFCorpus → data/
uv run python scripts/build_index.py             # index general embedder in Qdrant
uv run python scripts/eval_retrieval.py          # reproduce the retrieval table

uv run ask "What are the cardiovascular benefits of the Mediterranean diet?"
uv run streamlit run app.py                      # Ask + Benchmark UI
```

Quality gates (also enforced in CI):

```bash
uv run ruff check . && uv run mypy && uv run pytest
```

## Deploying

The app runs on a free **Hugging Face Space** (Docker SDK, running
Streamlit). It self-bootstraps the corpus on first boot and reads its
vectors from your Qdrant Cloud cluster. Set three Space secrets —
`GOOGLE_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` — and push. See
[DEPLOY.md](DEPLOY.md) for the step-by-step.

## What's deliberately not here

- **No fine-tuning** — off-the-shelf embedders/rerankers only.
- **No agentic orchestration** — biorag is a retrieval pipeline, not an agent.
- **No multi-provider LLM abstraction** — Gemini only.
- **No GPU requirement** — CPU-friendly models throughout.
- **Not a medical-advice tool** — cited literature Q&A only.

## Repo layout

```
src/biorag/
  config.py        env/settings (pydantic-settings)
  corpus.py        NFCorpus loader + normalized on-disk form
  chunk.py         document chunking
  embed.py         Embedder protocol, general/biomedical models, disk cache
  index.py         Qdrant collection build + upsert
  retrieve.py      Retriever protocol + DenseRetriever
  bm25.py          BM25 lexical retriever
  hybrid.py        Reciprocal Rank Fusion + HybridRetriever
  rerank.py        cross-encoder reranking stage
  rewrite.py       HyDE query rewriting (Gemini)
  generate.py      grounded answer generation + citation parsing
  cli.py           `ask` console script
  eval/            retrieval + answer-quality harnesses
scripts/           build_corpus, build_index, eval_retrieval, eval_answers
tests/             unit + end-to-end integration tests
app.py             Streamlit Ask + Benchmark UI
```
