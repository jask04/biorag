"""Biomedical corpus + evaluation dataset loaders.

The primary dataset is **BEIR NFCorpus** (a small biomedical IR benchmark
shipping a corpus, queries, and gold relevance judgments). This module:

1. Downloads NFCorpus from Hugging Face and writes it to ``data/nfcorpus/``
   in a normalized on-disk form decoupled from the upstream schema.
2. Exposes typed loaders that read those files back into plain dataclasses.

The on-disk normalized layout is:

    data/nfcorpus/
      corpus.jsonl   # one document per line: {"id", "title", "text"}
      queries.jsonl  # one query per line:    {"id", "text"}
      qrels.tsv      # tab-separated: query_id, doc_id, relevance (header row)

Downloads land under ``data/`` which is gitignored. The script
``scripts/build_corpus.py`` invokes the downloader and prints stats.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

DATA_DIR: Final[Path] = Path("data") / "nfcorpus"
CORPUS_FILE: Final[str] = "corpus.jsonl"
QUERIES_FILE: Final[str] = "queries.jsonl"
QRELS_FILE: Final[str] = "qrels.tsv"

# Upstream BEIR identifiers on the Hugging Face hub.
_HF_CORPUS_REPO: Final[str] = "BeIR/nfcorpus"
_HF_QRELS_REPO: Final[str] = "BeIR/nfcorpus-qrels"


@dataclass(frozen=True, slots=True)
class Document:
    """A single corpus document (title + abstract for NFCorpus)."""

    id: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class Query:
    """A single information-need query."""

    id: str
    text: str


@dataclass(frozen=True, slots=True)
class QrelEntry:
    """A gold relevance judgment: ``(query, doc) -> graded relevance``."""

    query_id: str
    doc_id: str
    relevance: int


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Summary statistics for a downloaded normalized corpus."""

    documents: int
    queries: int
    qrels: int
    relevant_pairs: int  # qrel rows with relevance > 0

    def render(self) -> str:
        return (
            f"documents:      {self.documents:>7}\n"
            f"queries:        {self.queries:>7}\n"
            f"qrels rows:     {self.qrels:>7}\n"
            f"relevant pairs: {self.relevant_pairs:>7}"
        )


# ---------- Writers (used by the downloader) ----------


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------- Loaders (typed) ----------


def iter_documents(path: Path = DATA_DIR / CORPUS_FILE) -> Iterator[Document]:
    """Stream documents from ``corpus.jsonl``."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            yield Document(id=row["id"], title=row["title"], text=row["text"])


def iter_queries(path: Path = DATA_DIR / QUERIES_FILE) -> Iterator[Query]:
    """Stream queries from ``queries.jsonl``."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            yield Query(id=row["id"], text=row["text"])


def iter_qrels(path: Path = DATA_DIR / QRELS_FILE) -> Iterator[QrelEntry]:
    """Stream qrel rows from ``qrels.tsv`` (skipping the header)."""
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline()
        if not header.startswith("query_id"):
            raise ValueError(f"unexpected qrels header in {path}: {header!r}")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, did, rel = line.split("\t")
            yield QrelEntry(query_id=qid, doc_id=did, relevance=int(rel))


def ensure_corpus(out_dir: Path = DATA_DIR) -> Path:
    """Download the corpus if the normalized files are missing.

    Lets a fresh environment (e.g. a Hugging Face Space on first boot)
    self-bootstrap without a separate build step — if ``corpus.jsonl``
    already exists it's a no-op.
    """
    if (out_dir / CORPUS_FILE).exists():
        return out_dir
    download_nfcorpus(out_dir)
    return out_dir


def load_documents(path: Path = DATA_DIR / CORPUS_FILE) -> list[Document]:
    return list(iter_documents(path))


def load_queries(path: Path = DATA_DIR / QUERIES_FILE) -> list[Query]:
    return list(iter_queries(path))


def load_qrels(path: Path = DATA_DIR / QRELS_FILE) -> list[QrelEntry]:
    return list(iter_qrels(path))


def compute_stats(out_dir: Path = DATA_DIR) -> CorpusStats:
    """Compute :class:`CorpusStats` from a normalized directory on disk."""
    docs = sum(1 for _ in iter_documents(out_dir / CORPUS_FILE))
    queries = sum(1 for _ in iter_queries(out_dir / QUERIES_FILE))
    qrels_rows = list(iter_qrels(out_dir / QRELS_FILE))
    return CorpusStats(
        documents=docs,
        queries=queries,
        qrels=len(qrels_rows),
        relevant_pairs=sum(1 for q in qrels_rows if q.relevance > 0),
    )


# ---------- Downloader ----------


def download_nfcorpus(
    out_dir: Path = DATA_DIR,
    qrels_split: str = "test",
) -> CorpusStats:
    """Download BEIR NFCorpus from HF and write the normalized files.

    Args:
        out_dir: Target directory. Created if missing.
        qrels_split: Which qrels split to use (``"train"|"dev"|"test"``).
            ``"test"`` is the standard BEIR evaluation split.

    Returns:
        :class:`CorpusStats` for the written files.
    """
    # Imported lazily so ``from biorag.corpus import Document`` stays cheap.
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_ds = load_dataset(_HF_CORPUS_REPO, "corpus", split="corpus")
    _write_jsonl(
        out_dir / CORPUS_FILE,
        (
            {
                "id": str(row["_id"]),
                "title": str(row.get("title") or ""),
                "text": str(row.get("text") or ""),
            }
            for row in corpus_ds
        ),
    )

    queries_ds = load_dataset(_HF_CORPUS_REPO, "queries", split="queries")
    _write_jsonl(
        out_dir / QUERIES_FILE,
        (
            {"id": str(row["_id"]), "text": str(row.get("text") or "")}
            for row in queries_ds
        ),
    )

    qrels_ds = load_dataset(_HF_QRELS_REPO, split=qrels_split)
    qrels_path = out_dir / QRELS_FILE
    with qrels_path.open("w", encoding="utf-8") as fh:
        fh.write("query_id\tdoc_id\trelevance\n")
        for row in qrels_ds:
            fh.write(
                f"{row['query-id']}\t{row['corpus-id']}\t{int(row['score'])}\n"
            )

    return compute_stats(out_dir)


__all__ = [
    "CORPUS_FILE",
    "DATA_DIR",
    "QRELS_FILE",
    "QUERIES_FILE",
    "CorpusStats",
    "Document",
    "QrelEntry",
    "Query",
    "compute_stats",
    "download_nfcorpus",
    "ensure_corpus",
    "iter_documents",
    "iter_qrels",
    "iter_queries",
    "load_documents",
    "load_qrels",
    "load_queries",
]
