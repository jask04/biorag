"""Tests for HyDE rewriting, the disk cache, and the composition wrapper.

The real :class:`GeminiHyDERewriter` is exercised by the eval harness
when it's run with HyDE configs. These tests pin down the cache, the
key-sanitization defense-in-depth, and the :class:`HydeRetriever`
composition using a deterministic fake rewriter — fast and offline.
"""

from __future__ import annotations

from pathlib import Path

from biorag.retrieve import RetrievalResult
from biorag.rewrite import (
    HydeRetriever,
    RewriteCache,
    Rewriter,
    _sanitize,
)


class CountingFakeRewriter:
    """Deterministic rewrite, counts how many times it's called."""

    def __init__(self) -> None:
        self.name = "fake-rewriter"
        self.calls: list[str] = []

    def rewrite(self, query: str) -> str:
        self.calls.append(query)
        return f"hypothetical passage about {query}"


class FakeBase:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        self.calls.append((query, k))
        return [
            RetrievalResult(
                doc_id="D1", score=0.9, chunk_id="D1#0", title="title-D1"
            )
        ]


def test_fake_rewriter_satisfies_protocol() -> None:
    assert isinstance(CountingFakeRewriter(), Rewriter)


def test_hyde_retriever_passes_rewritten_query_to_base() -> None:
    base = FakeBase()
    rewriter = CountingFakeRewriter()
    hits = HydeRetriever(base, rewriter).retrieve("does vitamin C help?", k=5)
    assert rewriter.calls == ["does vitamin C help?"]
    assert base.calls == [("hypothetical passage about does vitamin C help?", 5)]
    assert hits[0].doc_id == "D1"


def test_hyde_retriever_falls_back_to_original_on_empty_rewrite() -> None:
    class EmptyRewriter:
        name = "empty"

        def rewrite(self, query: str) -> str:
            return ""

    base = FakeBase()
    HydeRetriever(base, EmptyRewriter()).retrieve("orig query", k=3)
    assert base.calls == [("orig query", 3)]


def test_rewrite_cache_round_trips_through_disk(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    cache = RewriteCache(cache_path)
    cache.put("k1", "first passage")
    cache.put("k2", "second passage")

    revived = RewriteCache(cache_path)
    assert len(revived) == 2
    assert revived.get("k1") == "first passage"
    assert "k2" in revived


def test_rewrite_cache_persists_each_put_immediately(tmp_path: Path) -> None:
    # Simulates a mid-eval crash: every put must be on disk before the next.
    path = tmp_path / "c.jsonl"
    RewriteCache(path).put("k", "v")
    assert path.read_text(encoding="utf-8").strip() != ""


def test_sanitize_redacts_api_key_shapes() -> None:
    dirty = "Auth error using key=AIzaSyA1B2c3D4E5_F-XY for project X"
    cleaned = _sanitize(dirty)
    assert "AIzaSyA1B2" not in cleaned
    assert "AIza<REDACTED>" in cleaned


def test_cache_put_sanitizes_text_on_write(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    RewriteCache(path).put("k", "embedded key AIzaSyAaaaaaaaaa here")
    written = path.read_text(encoding="utf-8")
    assert "AIzaSyAaaaaa" not in written
    assert "AIza<REDACTED>" in written
