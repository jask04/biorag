"""Tests for the embedder protocol and on-disk cache.

The real :class:`SentenceTransformerEmbedder` is exercised by the eval
harness in later days. These tests pin down the protocol contract and the
cache semantics using a deterministic in-process fake — fast and offline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from biorag.embed import (
    CachedEmbedder,
    Embedder,
    EmbeddingCache,
    EmbeddingMatrix,
)

DIM = 4


class CountingFakeEmbedder:
    """Deterministic per-text fake; counts how many times `embed` is called."""

    def __init__(self) -> None:
        self.name = "fake-embedder"
        self.dimension = DIM
        self.calls = 0
        self.texts_seen: list[str] = []

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
        self.calls += 1
        self.texts_seen.extend(texts)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = (
                int.from_bytes(hashlib.sha1(t.encode("utf-8")).digest()[:4], "big")
                % (2**32)
            )
            out[i] = np.random.default_rng(seed).random(DIM).astype(np.float32)
        return out


def test_fake_satisfies_embedder_protocol() -> None:
    assert isinstance(CountingFakeEmbedder(), Embedder)


def test_cached_embedder_serves_repeats_without_calling_inner(tmp_path: Path) -> None:
    inner = CountingFakeEmbedder()
    cached = CachedEmbedder(inner, cache_dir=tmp_path)

    first = cached.embed(["a", "b", "c"])
    assert first.shape == (3, DIM)
    assert inner.calls == 1
    assert inner.texts_seen == ["a", "b", "c"]

    again = cached.embed(["a", "b", "c"])
    np.testing.assert_array_equal(first, again)
    assert inner.calls == 1  # nothing new — fully cache-served


def test_cached_embedder_only_re_embeds_misses(tmp_path: Path) -> None:
    inner = CountingFakeEmbedder()
    cached = CachedEmbedder(inner, cache_dir=tmp_path)

    cached.embed(["a", "b"])
    assert inner.calls == 1
    assert inner.texts_seen == ["a", "b"]

    mixed = cached.embed(["a", "c", "b", "d"])
    # Only the two new texts ("c", "d") should reach the inner model.
    assert inner.calls == 2
    assert inner.texts_seen[-2:] == ["c", "d"]
    # Returned matrix still has one row per input, in input order.
    assert mixed.shape == (4, DIM)


def test_cache_round_trips_through_disk(tmp_path: Path) -> None:
    inner = CountingFakeEmbedder()
    cached = CachedEmbedder(inner, cache_dir=tmp_path)
    cached.embed(["alpha", "beta"])
    cached.save()

    inner2 = CountingFakeEmbedder()
    revived = CachedEmbedder(inner2, cache_dir=tmp_path)
    revived.embed(["alpha", "beta"])
    assert inner2.calls == 0


def test_empty_input_does_not_call_inner(tmp_path: Path) -> None:
    inner = CountingFakeEmbedder()
    cached = CachedEmbedder(inner, cache_dir=tmp_path)
    result = cached.embed([])
    assert result.shape == (0, DIM)
    assert inner.calls == 0


def test_put_rejects_wrong_shape(tmp_path: Path) -> None:
    cache = EmbeddingCache("m", dimension=DIM, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="vector shape"):
        cache.put("hello", np.zeros((DIM + 1,), dtype=np.float32))


def test_inner_mismatch_dimension_is_caught(tmp_path: Path) -> None:
    class WrongDim:
        name = "wrong"
        dimension = DIM

        def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
            return np.zeros((len(texts), DIM + 1), dtype=np.float32)

    cached = CachedEmbedder(WrongDim(), cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="inner embedder returned shape"):
        cached.embed(["x"])
