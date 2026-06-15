"""Grounded answer generation with inline citations.

Takes the top-k retrieved documents from any :class:`Retriever`, hands the
``(query, passages)`` pair to Gemini under a strict system prompt that
allows answers only from the supplied passages, and returns an
:class:`Answer` carrying the model's prose plus structured citations
linked back to the documents.

Two ideas are worth surfacing:

1. **Inline citations are part of the contract, not decoration.** The
   prompt requires every factual claim to be tagged ``[doc_id]``; the
   parser then validates the cited ids against the actually-retrieved
   set and drops the rest, so a hallucinated id never leaks into the
   structured ``citations`` field.

2. **Explicit "unsupported" path.** When the passages don't answer the
   question the model is instructed to emit a fixed sentinel; callers
   can flip on :attr:`Answer.unsupported` rather than rely on heuristic
   string sniffing.

The Gemini key is read only inside the SDK client constructor and never
logged; the same ``AIza``-scrubber from :mod:`biorag.rewrite` runs on
every generated text and error message as defense in depth.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

from biorag.config import get_settings
from biorag.corpus import Document
from biorag.retrieve import RetrievalResult
from biorag.rewrite import _is_retryable, _sanitize  # reuse defense-in-depth

DEFAULT_MODEL: Final[str] = "gemini-2.5-flash-lite"
DEFAULT_K: Final[int] = 5
DEFAULT_MAX_PASSAGE_CHARS: Final[int] = 1200
DEFAULT_MAX_RETRIES: Final[int] = 5
DEFAULT_BACKOFF_BASE: Final[float] = 2.0
UNSUPPORTED_MARKER: Final[str] = "The provided passages do not address this question."

# Matches both single ids ``[MED-123]`` and comma-joined groups
# ``[MED-123, MED-456]``. Capture the inner content; split on comma.
_CITATION_GROUP_RE: Final[re.Pattern[str]] = re.compile(
    r"\[([A-Za-z][A-Za-z0-9]*-\d+(?:\s*,\s*[A-Za-z][A-Za-z0-9]*-\d+)*)\]"
)

SYSTEM_PROMPT: Final[str] = (
    "You are a biomedical literature retrieval assistant. Answer the "
    "user's question STRICTLY from the passages below. Each passage is "
    "labeled with a document id in square brackets.\n\n"
    "Rules:\n"
    "1. Cite every factual claim by appending the relevant document "
    'id(s) in square brackets, e.g. "Vitamin C reduces cold duration '
    '[MED-123]." Multiple citations may be combined: [MED-123, MED-456].\n'
    "2. If the passages do not contain enough information to answer the "
    f'question, respond EXACTLY with: "{UNSUPPORTED_MARKER}" Then stop.\n'
    "3. Do not introduce facts that are not supported by the passages. "
    "Do not speculate. Do not provide medical advice — this is a "
    "literature retrieval tool, not a clinical aid.\n"
    "4. Keep your answer concise (2-4 sentences).\n\n"
    "PASSAGES:\n{context}\n\n"
    "QUESTION: {query}\n\n"
    "ANSWER:"
)


@dataclass(frozen=True, slots=True)
class Citation:
    """A document referenced by the generated answer."""

    doc_id: str
    title: str


@dataclass(frozen=True, slots=True)
class Answer:
    """A grounded answer plus its provenance."""

    query: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievalResult] = field(default_factory=list)

    @property
    def unsupported(self) -> bool:
        """True when the model declined to answer for lack of evidence."""
        return UNSUPPORTED_MARKER.lower() in self.text.lower()


def format_passages(
    hits: Sequence[RetrievalResult],
    corpus: Mapping[str, Document],
    max_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
) -> str:
    """Render retrieved hits as the labeled-passage block for the prompt.

    Document text is truncated to ``max_chars`` per passage so a single
    long abstract can't crowd out the others. Truncation is character-
    based (cheap, deterministic) rather than token-based — Gemini's
    context window comfortably absorbs the difference.
    """
    blocks: list[str] = []
    for hit in hits:
        doc = corpus.get(hit.doc_id)
        title = (doc.title if doc else hit.title).strip()
        text = (doc.text if doc else "").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        body = f"{title}\n{text}" if title and text else title or text
        blocks.append(f"[{hit.doc_id}] {body}")
    return "\n\n".join(blocks)


def extract_citations(
    text: str, allowed_ids: Sequence[str]
) -> list[str]:
    """Pull cited doc ids out of the answer text, in first-appearance order.

    Filters to ``allowed_ids`` so an id the model invented (not in the
    retrieved set) is silently dropped from the structured field, even
    if it still appears in the prose.
    """
    allowed = set(allowed_ids)
    seen: dict[str, None] = {}
    for match in _CITATION_GROUP_RE.finditer(text):
        for raw in match.group(1).split(","):
            doc_id = raw.strip()
            if doc_id in allowed and doc_id not in seen:
                seen[doc_id] = None
    return list(seen)


def build_citations(
    doc_ids: Sequence[str],
    corpus: Mapping[str, Document],
    hits: Sequence[RetrievalResult],
) -> list[Citation]:
    """Lift cited doc ids into ``Citation`` objects with their titles."""
    titles_by_id: dict[str, str] = {hit.doc_id: hit.title for hit in hits}
    citations: list[Citation] = []
    for doc_id in doc_ids:
        doc = corpus.get(doc_id)
        title = (doc.title if doc else titles_by_id.get(doc_id, "")).strip()
        citations.append(Citation(doc_id=doc_id, title=title))
    return citations


@runtime_checkable
class AnswerGenerator(Protocol):
    """Turns a query plus retrieved hits into a grounded :class:`Answer`."""

    name: str

    def answer(
        self, query: str, hits: Sequence[RetrievalResult]
    ) -> Answer: ...


class GeminiAnswerGenerator:
    """Gemini-backed answerer that emits citation-grounded prose."""

    def __init__(
        self,
        corpus: Mapping[str, Document],
        model: str = DEFAULT_MODEL,
        prompt: str = SYSTEM_PROMPT,
        max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        api_key: str | None = None,
    ) -> None:
        from google import genai

        # ``api_key`` lets the UI inject a visitor's own key (BYOK) without
        # touching the process environment. Falls back to the configured
        # key for the CLI and the eval harness.
        resolved_key = api_key or get_settings().google_api_key
        if not resolved_key:
            raise RuntimeError(
                "No Gemini API key available — set GOOGLE_API_KEY in .env or "
                "pass api_key explicitly"
            )
        # Key is consumed only here, never stored on self or logged.
        self._client = genai.Client(api_key=resolved_key)
        self._corpus = corpus
        self._model = model
        self._prompt = prompt
        self._max_passage_chars = max_passage_chars
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self.name = f"gemini-{model}"

    def answer(
        self, query: str, hits: Sequence[RetrievalResult]
    ) -> Answer:
        if not hits:
            return Answer(
                query=query,
                text=UNSUPPORTED_MARKER,
                citations=[],
                retrieved=list(hits),
            )
        passages = format_passages(
            hits, self._corpus, max_chars=self._max_passage_chars
        )
        contents = self._prompt.format(context=passages, query=query)
        raw = self._call_with_backoff(contents)
        text = _sanitize(raw).strip()
        cited_ids = extract_citations(text, [h.doc_id for h in hits])
        citations = build_citations(cited_ids, self._corpus, hits)
        return Answer(
            query=query,
            text=text,
            citations=citations,
            retrieved=list(hits),
        )

    def _call_with_backoff(self, contents: str) -> str:
        last_err = ""
        for attempt in range(self._max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                )
            except Exception as exc:  # noqa: BLE001 — we sanitize and re-raise
                msg = _sanitize(str(exc))
                last_err = msg
                if _is_retryable(msg) and attempt < self._max_retries - 1:
                    time.sleep(self._backoff_base ** (attempt + 1))
                    continue
                raise RuntimeError(f"Gemini call failed: {msg}") from None
            text = getattr(response, "text", None)
            if not text:
                # Empty (safety filter, blocked content). Treat as
                # unsupported rather than failing the caller.
                return UNSUPPORTED_MARKER
            return str(text)
        raise RuntimeError(
            f"Gemini call failed after {self._max_retries} retries: {last_err}"
        )


__all__ = [
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_K",
    "DEFAULT_MAX_PASSAGE_CHARS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MODEL",
    "SYSTEM_PROMPT",
    "UNSUPPORTED_MARKER",
    "Answer",
    "AnswerGenerator",
    "Citation",
    "GeminiAnswerGenerator",
    "build_citations",
    "extract_citations",
    "format_passages",
]
