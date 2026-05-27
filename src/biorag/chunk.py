"""Document chunking.

NFCorpus documents are short (title + abstract), so most documents become
a single chunk. Anything longer than ``max_words`` is split into
overlapping word-windows. Chunks carry their source ``doc_id`` so a
retrieved chunk can always be attributed back to the original document.

We chunk on whitespace tokens, not subword tokens. Day 3 doesn't depend on
a tokenizer choice, and the BEIR docs we care about are abstract-length;
the embedder later truncates to its own max sequence length internally.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from biorag.corpus import Document

DEFAULT_MAX_WORDS = 256
DEFAULT_OVERLAP = 32


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text tied back to its source document."""

    id: str
    doc_id: str
    text: str


def _document_body(doc: Document) -> str:
    """Concatenate title + text so the title is embedded with the body."""
    title = doc.title.strip()
    text = doc.text.strip()
    if title and text:
        return f"{title}. {text}"
    return title or text


def chunk_document(
    doc: Document,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split a document into one or more overlapping word-window chunks.

    Args:
        doc: Source document.
        max_words: Maximum whitespace tokens per chunk.
        overlap: How many tokens the next window re-includes from the prior
            window. Must satisfy ``0 <= overlap < max_words``.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap < max_words:
        raise ValueError("overlap must be in [0, max_words)")

    body = _document_body(doc)
    if not body:
        return []

    words = body.split()
    if len(words) <= max_words:
        return [Chunk(id=f"{doc.id}#0", doc_id=doc.id, text=body)]

    chunks: list[Chunk] = []
    step = max_words - overlap
    i = 0
    start = 0
    while start < len(words):
        window = words[start : start + max_words]
        chunks.append(
            Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, text=" ".join(window))
        )
        if start + max_words >= len(words):
            break
        start += step
        i += 1
    return chunks


def chunk_documents(
    docs: Iterable[Document],
    max_words: int = DEFAULT_MAX_WORDS,
    overlap: int = DEFAULT_OVERLAP,
) -> Iterator[Chunk]:
    """Stream chunks from an iterable of documents."""
    for doc in docs:
        yield from chunk_document(doc, max_words=max_words, overlap=overlap)


__all__ = [
    "DEFAULT_MAX_WORDS",
    "DEFAULT_OVERLAP",
    "Chunk",
    "chunk_document",
    "chunk_documents",
]
