"""Answer-quality evaluation with ragas under a tight free-tier budget.

The retrieval eval (:mod:`biorag.eval.retrieval`) measures whether the
right documents come back. This module measures whether the *answers*
built on top of those documents are faithful to the retrieved evidence
and on-topic for the question.

Free-tier constraint shapes everything here:

* A small deterministic sample (default ``N=3``) — the first ``N``
  NFCorpus qrel-positive queries, in queries.jsonl order, so the sample
  is reproducible and grows monotonically with ``N``.
* Two metrics by default: **faithfulness** (do the answer's claims
  follow from the retrieved context?) and **answer_relevancy** (does
  the answer actually address the question?). ``context_precision`` is
  available as an opt-in but skipped by default because it costs
  ``k`` LLM calls per question.
* **LangChain SQLite cache** for every Gemini call. A run that hits the
  daily quota partway through saves everything it computed; the next
  day's run resumes from cache and only spends quota on the new work.
* **Local BGE-small embeddings** as the ragas embedding model so we
  don't burn embedding-API quota on top of the LLM-judge calls.

Output is a small JSON record per pipeline configuration: which sample
was used, the generated answer for each question, and the per-question
ragas scores plus the aggregate means. That single file is what the
Day 14 README headline reads from.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from biorag.config import get_settings
from biorag.corpus import Document, QrelEntry, Query
from biorag.generate import Answer, AnswerGenerator
from biorag.retrieve import Retriever

DEFAULT_RESULTS_DIR: Final[Path] = Path("eval_results")
DEFAULT_CACHE_DIR: Final[Path] = Path(".cache")
DEFAULT_SAMPLE_SIZE: Final[int] = 3
DEFAULT_K: Final[int] = 3
DEFAULT_RAGAS_MODEL: Final[str] = "gemini-2.5-flash-lite"


@dataclass(frozen=True, slots=True)
class SampleItem:
    """One scored query in the answer-eval sample, with its gold qrels."""

    query: Query
    relevant_doc_ids: list[str]


@dataclass(slots=True)
class AnsweredItem:
    """A :class:`SampleItem` after the pipeline has produced an answer."""

    query_id: str
    question: str
    answer: str
    citations: list[str]
    contexts: list[str]
    retrieved_doc_ids: list[str]
    unsupported: bool
    relevant_doc_ids: list[str]


@dataclass(slots=True)
class AnswerEvalResult:
    """Per-config ragas output + the per-question records that produced it."""

    config: str
    sample_size: int
    metrics: dict[str, float] = field(default_factory=dict)
    per_question: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def select_sample(
    queries: Sequence[Query],
    qrels: Sequence[QrelEntry],
    n: int,
) -> list[SampleItem]:
    """Pick the first ``n`` qrel-positive queries in queries.jsonl order.

    Deterministic — given the same NFCorpus snapshot, every run picks the
    same sample. Growing ``n`` only appends; earlier indices stay stable
    so cached LLM calls keep hitting on re-runs.
    """
    relevance_by_qid: dict[str, dict[str, int]] = {}
    for entry in qrels:
        relevance_by_qid.setdefault(entry.query_id, {})[entry.doc_id] = entry.relevance

    selected: list[SampleItem] = []
    for query in queries:
        graded = relevance_by_qid.get(query.id)
        if not graded:
            continue
        positives = [doc_id for doc_id, rel in graded.items() if rel > 0]
        if not positives:
            continue
        selected.append(SampleItem(query=query, relevant_doc_ids=sorted(positives)))
        if len(selected) >= n:
            break
    return selected


def passages_for(
    answer: Answer, corpus: dict[str, Document], max_chars: int
) -> list[str]:
    """The retrieved doc texts that the answer was conditioned on.

    Used as ragas' ``contexts`` column. We rebuild from the corpus map so
    truncation matches what the answer-generation prompt actually saw.
    """
    chunks: list[str] = []
    for hit in answer.retrieved:
        doc = corpus.get(hit.doc_id)
        text = (doc.text if doc else "").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        title = (doc.title if doc else hit.title).strip()
        chunks.append(f"{title}. {text}".strip(". "))
    return chunks


def answer_question(
    item: SampleItem,
    retriever: Retriever,
    generator: AnswerGenerator,
    corpus: dict[str, Document],
    k: int,
    max_passage_chars: int,
) -> AnsweredItem:
    """Run the retrieve→generate path once for a single sample item."""
    hits = retriever.retrieve(item.query.text, k=k)
    answer = generator.answer(item.query.text, hits)
    return AnsweredItem(
        query_id=item.query.id,
        question=item.query.text,
        answer=answer.text,
        citations=[c.doc_id for c in answer.citations],
        contexts=passages_for(answer, corpus, max_chars=max_passage_chars),
        retrieved_doc_ids=[h.doc_id for h in answer.retrieved],
        unsupported=answer.unsupported,
        relevant_doc_ids=item.relevant_doc_ids,
    )


# ---------- ragas wiring ----------


def install_langchain_sqlite_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Globally route every LangChain LLM call through a SQLite cache.

    Critical for free-tier reproducibility: any call that succeeded
    yesterday is served from disk today, so a re-run only spends quota
    on questions that haven't been scored yet.
    """
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache

    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "langchain_llm_cache.db"
    set_llm_cache(SQLiteCache(database_path=str(db_path)))
    return db_path


def _build_ragas_clients(model: str) -> tuple[Any, Any]:
    """LangChain Gemini + local-embedding wrapper, both wrapped for ragas."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from biorag.embed import CachedEmbedder, general_embedder

    settings = get_settings()
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY must be set in .env for the ragas LLM judge"
        )
    chat = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        temperature=0.0,
    )
    embedder = CachedEmbedder(general_embedder())
    return (
        LangchainLLMWrapper(chat),
        LangchainEmbeddingsWrapper(_LocalEmbeddingShim(embedder)),
    )


class _LocalEmbeddingShim:
    """LangChain-compatible embeddings backed by our :class:`CachedEmbedder`.

    Avoids burning Gemini's separate embedding-API quota on ragas runs —
    BGE-small is already cached on disk from the retrieval eval.
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._embedder.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _build_dataset(answered: Sequence[AnsweredItem]) -> Any:
    """Compose the ragas-shaped HF dataset from answered items."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "question": [a.question for a in answered],
            "answer": [a.answer for a in answered],
            "contexts": [a.contexts for a in answered],
            "reference": [a.answer for a in answered],
        }
    )


def score_with_ragas(
    answered: Sequence[AnsweredItem],
    *,
    model: str = DEFAULT_RAGAS_MODEL,
    include_context_precision: bool = False,
) -> tuple[dict[str, float], list[dict[str, float]], list[str]]:
    """Run ragas faithfulness + answer_relevancy (and optionally context_precision).

    Returns ``(aggregate_means, per_question_scores, notes)`` where
    ``notes`` captures soft failures (e.g. quota-exhaustion that left
    some rows unscored) so the commit record is honest.
    """
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    notes: list[str] = []
    metrics: list[Any] = [faithfulness, answer_relevancy]
    if include_context_precision:
        metrics.append(context_precision)

    llm, embeddings = _build_ragas_clients(model)
    dataset = _build_dataset(answered)
    try:
        # ragas' type stubs declare a union return that includes Executor;
        # at runtime we always get an EvaluationResult here. Cast to Any so
        # mypy stops second-guessing the attribute access.
        result: Any = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )
    except Exception as exc:  # noqa: BLE001 — surface as soft failure
        notes.append(f"ragas.evaluate raised: {type(exc).__name__}: {exc}")
        return {}, [], notes

    aggregate = {k: float(v) for k, v in result._repr_dict.items()}
    df = result.to_pandas()
    _ragas_input_cols = {"user_input", "response", "retrieved_contexts", "reference"}
    metric_cols = [c for c in df.columns if c not in _ragas_input_cols]
    per_question = [
        {col: float(row[col]) for col in metric_cols if row[col] == row[col]}
        for _, row in df.iterrows()
    ]
    return aggregate, per_question, notes


def evaluate_pipeline(
    config_name: str,
    sample: Sequence[SampleItem],
    retriever: Retriever,
    generator: AnswerGenerator,
    corpus: dict[str, Document],
    *,
    k: int = DEFAULT_K,
    max_passage_chars: int = 1200,
    include_context_precision: bool = False,
    model: str = DEFAULT_RAGAS_MODEL,
) -> AnswerEvalResult:
    """Generate answers for ``sample`` and score them with ragas."""
    answered: list[AnsweredItem] = []
    notes: list[str] = []
    for idx, sample_item in enumerate(sample, start=1):
        try:
            answered.append(
                answer_question(
                    sample_item, retriever, generator, corpus, k, max_passage_chars
                )
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(
                f"answer generation stopped at q{idx}/{len(sample)}: "
                f"{type(exc).__name__}: {exc}"
            )
            break
        # Light throttle to be polite to the free-tier RPM ceiling.
        time.sleep(0.5)

    aggregate, per_question, ragas_notes = score_with_ragas(
        answered,
        model=model,
        include_context_precision=include_context_precision,
    )
    notes.extend(ragas_notes)

    per_record: list[dict[str, Any]] = []
    for i, answered_item in enumerate(answered):
        record = asdict(answered_item)
        if i < len(per_question):
            record["scores"] = per_question[i]
        per_record.append(record)

    return AnswerEvalResult(
        config=config_name,
        sample_size=len(answered),
        metrics=aggregate,
        per_question=per_record,
        notes=notes,
    )


def save_results(
    results: Sequence[AnswerEvalResult],
    *,
    path: Path = DEFAULT_RESULTS_DIR / "answers.json",
) -> Path:
    """Write the JSON record consumed by the Day 14 README headline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "config": r.config,
            "sample_size": r.sample_size,
            "metrics": r.metrics,
            "per_question": r.per_question,
            "notes": r.notes,
        }
        for r in results
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_K",
    "DEFAULT_RAGAS_MODEL",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_SAMPLE_SIZE",
    "AnswerEvalResult",
    "AnsweredItem",
    "SampleItem",
    "answer_question",
    "evaluate_pipeline",
    "install_langchain_sqlite_cache",
    "passages_for",
    "save_results",
    "score_with_ragas",
    "select_sample",
]
