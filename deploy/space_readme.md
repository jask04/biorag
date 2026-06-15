---
title: biorag
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
short_description: Cited biomedical Q&A with a benchmarked retrieval pipeline
---

# biorag

Evaluation-first biomedical RAG — ask cited questions over BEIR NFCorpus,
backed by a configurable retrieval pipeline (hybrid search, cross-encoder
reranking, HyDE) and a benchmark harness.

A cited literature Q&A assistant — **not** a medical-advice tool.

Source, full results table, and design notes:
**https://github.com/jask04/biorag**

This Space needs three secrets (Settings → Variables and secrets):
`GOOGLE_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`. Visitors can also paste
their own Gemini key in the sidebar.
