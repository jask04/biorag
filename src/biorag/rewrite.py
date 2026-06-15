"""HyDE query rewriting via Gemini.

Hypothetical Document Embeddings (Gao et al., 2022): instead of embedding
the user's terse query directly, ask an LLM to write a short hypothetical
passage that *would* answer the query, and embed that. The hypothetical
passage lives in the same prose distribution as the corpus, so its
embedding sits closer to relevant documents in cosine space than the
short interrogative form of the query does.

Gemini calls are cached to disk so a re-run of the eval harness over the
same 323 NFCorpus test queries doesn't re-pay the LLM cost — a hard
requirement on the free tier. The API key is read only from settings,
passed straight to the client, and never logged; errors and cached
artifacts are scrubbed of any accidentally embedded ``AIza...`` strings
as defense in depth.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from biorag.config import get_settings
from biorag.retrieve import RetrievalResult, Retriever

DEFAULT_CACHE_DIR: Final[Path] = Path(".cache") / "rewrites"
DEFAULT_MODEL: Final[str] = "gemini-2.5-flash-lite"
DEFAULT_MAX_RETRIES: Final[int] = 5
DEFAULT_BACKOFF_BASE: Final[float] = 2.0

# Scrub anything that looks like a Google API key from logged strings.
_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(r"AIza[0-9A-Za-z_\-]{10,}")

# Short and direct on purpose. NFCorpus queries are health/diet questions;
# we want the LLM to produce a plausible PubMed-abstract-flavored passage,
# not a hedged refusal.
HYDE_PROMPT: Final[str] = (
    "Write a short (2-3 sentence) hypothetical passage from a biomedical "
    "research article that would directly answer the question below. Use "
    "the technical vocabulary of biomedical literature (mechanisms, "
    "outcomes, study design). Do not refuse or hedge — produce a plausible "
    "passage even if you are not certain of the underlying facts.\n\n"
    "Question: {query}\n\nHypothetical passage:"
)


def _sanitize(text: str) -> str:
    """Strip anything that resembles a Google API key."""
    return _KEY_PATTERN.sub("AIza<REDACTED>", text)


def _hash_key(*parts: str) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


@runtime_checkable
class Rewriter(Protocol):
    """Anything that turns a query string into a rewritten query string."""

    name: str

    def rewrite(self, query: str) -> str: ...


class RewriteCache:
    """sha1-keyed text cache as a single JSONL file per model.

    JSONL gives us safe append-on-write semantics: a crash mid-eval
    doesn't lose the rewrites we already paid for.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._memory: dict[str, str] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    self._memory[str(row["key"])] = str(row["text"])

    def __len__(self) -> int:
        return len(self._memory)

    def __contains__(self, key: str) -> bool:
        return key in self._memory

    def get(self, key: str) -> str | None:
        return self._memory.get(key)

    def put(self, key: str, text: str) -> None:
        clean = _sanitize(text)
        self._memory[key] = clean
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "text": clean}) + "\n")


def _slugify(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


class GeminiHyDERewriter:
    """Generate a hypothetical answer passage with Gemini, with disk cache."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        prompt: str = HYDE_PROMPT,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        api_key: str | None = None,
    ) -> None:
        from google import genai

        # ``api_key`` supports BYOK from the UI; falls back to the configured
        # key for the CLI and eval harness.
        resolved_key = api_key or get_settings().google_api_key
        if not resolved_key:
            raise RuntimeError(
                "No Gemini API key available — set GOOGLE_API_KEY in .env or "
                "pass api_key explicitly"
            )
        # Key is consumed here and never stored on self or logged.
        self._client = genai.Client(api_key=resolved_key)
        self._model = model
        self._prompt = prompt
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self.name = f"hyde-{model}"
        self.cache = RewriteCache(cache_dir / f"{_slugify(model)}.jsonl")

    def rewrite(self, query: str) -> str:
        key = _hash_key(self._model, self._prompt, query)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        text = self._call_with_backoff(self._prompt.format(query=query))
        self.cache.put(key, text)
        return text

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
                # Empty response (safety filter, etc) — don't retry, just
                # return the empty string so the caller can move on.
                return ""
            return _sanitize(str(text)).strip()
        raise RuntimeError(
            f"Gemini call failed after {self._max_retries} retries: {last_err}"
        )


def _is_retryable(error_message: str) -> bool:
    lowered = error_message.lower()
    return (
        "429" in lowered
        or "resource_exhausted" in lowered
        or "rate limit" in lowered
        or "unavailable" in lowered
        or "503" in lowered
    )


class HydeRetriever:
    """Replace the query with its hypothetical-passage rewrite, then retrieve.

    Wraps any base :class:`Retriever`. For dense channels the rewrite shifts
    the query into corpus-prose distribution; for BM25 the rewrite trades
    short interrogative tokens for the LLM's domain vocabulary — usually a
    win on biomedical queries that are vocabulary-sparse on the user side.
    """

    def __init__(self, base: Retriever, rewriter: Rewriter) -> None:
        self._base = base
        self._rewriter = rewriter
        self.name = f"hyde+{getattr(base, 'name', type(base).__name__)}"

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        rewritten = self._rewriter.rewrite(query)
        if not rewritten:
            # Empty rewrite (e.g. safety filter) — fall back to the original.
            rewritten = query
        return self._base.retrieve(rewritten, k=k)


__all__ = [
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MODEL",
    "HYDE_PROMPT",
    "GeminiHyDERewriter",
    "HydeRetriever",
    "RewriteCache",
    "Rewriter",
]
