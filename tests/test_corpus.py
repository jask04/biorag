"""Round-trip tests for the corpus loaders against a fixture slice.

These tests do not touch Hugging Face — they exercise the on-disk schema
that the downloader writes, by constructing tiny fixture files in a
``tmp_path`` and reading them back through the typed loaders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biorag.corpus import (
    CORPUS_FILE,
    QRELS_FILE,
    QUERIES_FILE,
    Document,
    QrelEntry,
    Query,
    compute_stats,
    load_documents,
    load_qrels,
    load_queries,
)


@pytest.fixture
def fixture_dir(tmp_path: Path) -> Path:
    docs = [
        {"id": "MED-1", "title": "Vitamin C and the common cold",
         "text": "A short abstract about ascorbic acid trials."},
        {"id": "MED-2", "title": "Mediterranean diet outcomes",
         "text": "Observational evidence on cardiovascular endpoints."},
        {"id": "MED-3", "title": "Curcumin bioavailability",
         "text": "Pharmacokinetics of oral curcumin formulations."},
    ]
    queries = [
        {"id": "PLAIN-1", "text": "does vitamin C prevent colds"},
        {"id": "PLAIN-2", "text": "is the mediterranean diet heart-healthy"},
    ]
    qrels = [
        ("PLAIN-1", "MED-1", 2),
        ("PLAIN-1", "MED-3", 0),
        ("PLAIN-2", "MED-2", 1),
    ]

    (tmp_path / CORPUS_FILE).write_text(
        "\n".join(json.dumps(d) for d in docs) + "\n", encoding="utf-8"
    )
    (tmp_path / QUERIES_FILE).write_text(
        "\n".join(json.dumps(q) for q in queries) + "\n", encoding="utf-8"
    )
    (tmp_path / QRELS_FILE).write_text(
        "query_id\tdoc_id\trelevance\n"
        + "\n".join(f"{q}\t{d}\t{r}" for q, d, r in qrels)
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_load_documents(fixture_dir: Path) -> None:
    docs = load_documents(fixture_dir / CORPUS_FILE)
    assert len(docs) == 3
    assert docs[0] == Document(
        id="MED-1",
        title="Vitamin C and the common cold",
        text="A short abstract about ascorbic acid trials.",
    )


def test_load_queries(fixture_dir: Path) -> None:
    queries = load_queries(fixture_dir / QUERIES_FILE)
    assert queries == [
        Query(id="PLAIN-1", text="does vitamin C prevent colds"),
        Query(id="PLAIN-2", text="is the mediterranean diet heart-healthy"),
    ]


def test_load_qrels(fixture_dir: Path) -> None:
    qrels = load_qrels(fixture_dir / QRELS_FILE)
    assert qrels[0] == QrelEntry(query_id="PLAIN-1", doc_id="MED-1", relevance=2)
    assert {q.query_id for q in qrels} == {"PLAIN-1", "PLAIN-2"}


def test_qrels_rejects_missing_header(tmp_path: Path) -> None:
    bad = tmp_path / QRELS_FILE
    bad.write_text("PLAIN-1\tMED-1\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected qrels header"):
        load_qrels(bad)


def test_compute_stats(fixture_dir: Path) -> None:
    stats = compute_stats(fixture_dir)
    assert stats.documents == 3
    assert stats.queries == 2
    assert stats.qrels == 3
    assert stats.relevant_pairs == 2  # rows with relevance > 0
    rendered = stats.render()
    assert "documents:" in rendered
    assert "relevant pairs:" in rendered
