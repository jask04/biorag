"""Embedding models and an on-disk cache.

Defines an :class:`Embedder` protocol so the rest of the pipeline can
depend on a small interface (``name``, ``dimension``, ``embed``) and stay
ignorant of whether vectors come from a real model, a fake (in tests), or
a future remote API.

Two production implementations sit behind ``general_embedder()`` and
``biomedical_embedder()`` — the general-vs-domain comparison is one of the
headline eval results.

:class:`CachedEmbedder` wraps any ``Embedder`` with a sha1-keyed
persistent cache so re-runs of the eval harness don't re-pay the encode
cost. Cache files live under ``.cache/embeddings/`` (gitignored), one
``.npz`` per model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

EmbeddingMatrix = npt.NDArray[np.float32]

DEFAULT_CACHE_DIR = Path(".cache") / "embeddings"
GENERAL_MODEL = "BAAI/bge-small-en-v1.5"
BIOMEDICAL_MODEL = "NeuML/pubmedbert-base-embeddings"


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into a row-per-text float32 matrix."""

    name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix: ...


class SentenceTransformerEmbedder:
    """Adapter for any sentence-transformers model.

    Vectors are L2-normalized at encode time so dot product equals cosine
    similarity — important for both Qdrant ANN and BM25/dense fusion later.
    """

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        # Lazy import so importing ``biorag.embed`` is cheap and tests that
        # mock the embedder don't pay for torch.
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:  # pragma: no cover — sentence-transformers always sets it
            raise RuntimeError(f"could not determine embedding dim for {model_name}")
        self.dimension = int(dim)

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vecs = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr: EmbeddingMatrix = np.asarray(vecs, dtype=np.float32)
        return arr


def general_embedder() -> SentenceTransformerEmbedder:
    """The general-purpose baseline (BGE small)."""
    return SentenceTransformerEmbedder(GENERAL_MODEL)


def biomedical_embedder() -> SentenceTransformerEmbedder:
    """The domain-adapted variant (PubMedBERT)."""
    return SentenceTransformerEmbedder(BIOMEDICAL_MODEL)


# ---------- Cache ----------


def _slugify(model_name: str) -> str:
    return model_name.replace("/", "__")


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()


class EmbeddingCache:
    """sha1(text) → vector cache for one model, persisted as a single ``.npz``."""

    def __init__(
        self,
        model_name: str,
        dimension: int,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.path = cache_dir / f"{_slugify(model_name)}.npz"
        self._memory: dict[str, EmbeddingMatrix] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with np.load(self.path, allow_pickle=False) as data:
            keys = data["keys"]
            vectors = data["vectors"]
        for k, v in zip(keys.tolist(), vectors, strict=True):
            self._memory[str(k)] = np.asarray(v, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._memory)

    def __contains__(self, text: str) -> bool:
        return _hash_text(text) in self._memory

    def get(self, text: str) -> EmbeddingMatrix | None:
        return self._memory.get(_hash_text(text))

    def put(self, text: str, vector: EmbeddingMatrix) -> None:
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"vector shape {vector.shape} does not match cache dim "
                f"({self.dimension},)"
            )
        self._memory[_hash_text(text)] = vector.astype(np.float32, copy=False)

    def save(self) -> None:
        """Flush the cache to disk. Creates parent directories as needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._memory:
            # Still write an empty file so callers can tell save() ran.
            np.savez(
                self.path,
                keys=np.empty((0,), dtype="U40"),
                vectors=np.zeros((0, self.dimension), dtype=np.float32),
            )
            return
        keys = np.array(list(self._memory.keys()), dtype="U40")
        vectors = np.stack(list(self._memory.values())).astype(np.float32, copy=False)
        np.savez(self.path, keys=keys, vectors=vectors)


class CachedEmbedder:
    """Wraps an :class:`Embedder` with a persistent sha1-keyed cache.

    Misses are batched into a single underlying ``embed`` call so we keep
    the inner model's batching benefit.
    """

    def __init__(
        self,
        inner: Embedder,
        cache: EmbeddingCache | None = None,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self._inner = inner
        self.name = inner.name
        self.dimension = inner.dimension
        self._cache = cache or EmbeddingCache(
            inner.name, inner.dimension, cache_dir=cache_dir
        )

    @property
    def cache(self) -> EmbeddingCache:
        return self._cache

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self.dimension), dtype=np.float32)

        out: list[EmbeddingMatrix | None] = [self._cache.get(t) for t in text_list]
        miss_idx = [i for i, v in enumerate(out) if v is None]

        if miss_idx:
            miss_texts = [text_list[i] for i in miss_idx]
            fresh = self._inner.embed(miss_texts)
            if fresh.shape != (len(miss_texts), self.dimension):
                raise RuntimeError(
                    f"inner embedder returned shape {fresh.shape}; "
                    f"expected ({len(miss_texts)}, {self.dimension})"
                )
            for i, text, vec in zip(miss_idx, miss_texts, fresh, strict=True):
                self._cache.put(text, vec)
                out[i] = vec

        # Every slot is filled now.
        matrix = np.stack([v for v in out if v is not None])
        return matrix.astype(np.float32, copy=False)

    def save(self) -> None:
        self._cache.save()


__all__ = [
    "BIOMEDICAL_MODEL",
    "DEFAULT_CACHE_DIR",
    "GENERAL_MODEL",
    "CachedEmbedder",
    "Embedder",
    "EmbeddingCache",
    "EmbeddingMatrix",
    "SentenceTransformerEmbedder",
    "biomedical_embedder",
    "general_embedder",
]
