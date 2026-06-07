"""Integration tests for the M2 query pipeline — the acceptance criteria.

Covers: grounded answers carry valid citations to *retrieved* chunks; filters are
applied inside Qdrant; low-confidence and empty-filter queries refuse; fabricated
or out-of-range citation markers cannot survive; an unattributed answer is converted
to a refusal; a backend error surfaces as an error status (not a fake answer).

All against in-memory Qdrant with the offline embedder + extractive generator,
reusing the shared fixture corpus.
"""

from __future__ import annotations

import pytest

from research_navigator.config import Settings
from research_navigator.generate.generator import GeneratedAnswer, GenerationError
from research_navigator.generate.pipeline import AnswerStatus, QueryEngine
from research_navigator.ingest.embed import build_embedders
from research_navigator.ingest.pipeline import ingest_corpus
from research_navigator.ingest.qdrant_store import QdrantStore
from research_navigator.retrieve.filters import build_filter
from research_navigator.retrieve.hybrid import HybridRetriever
from research_navigator.retrieve.query import QueryAnalysis, analyze_query

pytestmark = pytest.mark.integration


def _ingested_engine(settings: Settings, store: QdrantStore) -> QueryEngine:
    ingest_corpus(settings, store=store)
    return QueryEngine(settings, store=store)


def _doc_ids(store: QdrantStore, settings: Settings) -> set[str]:
    points, _ = store.client.scroll(
        collection_name=settings.qdrant_collection, with_payload=["doc_id"], limit=256
    )
    return {str((p.payload or {})["doc_id"]) for p in points}


def test_answer_carries_valid_citations(settings: Settings, store: QdrantStore) -> None:
    settings.retrieval.refusal_min_score = 0.05  # offline cosine scale
    engine = _ingested_engine(settings, store)
    answer = engine.answer("How do transformers use the attention mechanism?")

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.citations, "an answered query must carry citations"
    # Every citation number actually appears as a marker in the text.
    for c in answer.citations:
        assert f"[{c.number}]" in answer.text
    # No fabricated citation: every cited doc is a real retrieved document.
    real = _doc_ids(store, settings)
    assert all(c.doc_id in real for c in answer.citations)


def test_low_confidence_refuses(settings: Settings, store: QdrantStore) -> None:
    settings.retrieval.refusal_min_score = 0.999  # nothing can clear this
    engine = _ingested_engine(settings, store)
    answer = engine.answer("How do transformers use attention?")
    assert answer.status is AnswerStatus.REFUSED
    assert answer.citations == []
    assert "enough relevant material" in answer.text


def test_empty_filter_refuses(settings: Settings, store: QdrantStore) -> None:
    settings.retrieval.refusal_min_score = 0.05
    engine = _ingested_engine(settings, store)
    # Corpus docs are from 2023; an in-1900 filter empties the candidate set.
    answer = engine.answer("transformers published in 1900")
    assert answer.analysis is not None
    assert answer.analysis.year_min == 1900
    assert answer.status is AnswerStatus.REFUSED


def test_filters_applied_inside_qdrant(settings: Settings, store: QdrantStore) -> None:
    ingest_corpus(settings, store=store)
    dense, sparse = build_embedders(settings)
    retriever = HybridRetriever(settings, store, dense, sparse)
    analysis = QueryAnalysis(raw_query="x", content_types=["survey_blog"])
    result = retriever.retrieve("prompting", query_filter=build_filter(analysis))
    assert result.chunks, "expected the survey to be retrievable"
    assert all(c.content_type == "survey_blog" for c in result.chunks)


def test_out_of_range_marker_is_stripped(settings: Settings, store: QdrantStore) -> None:
    settings.retrieval.refusal_min_score = 0.05
    ingest_corpus(settings, store=store)

    class _Stub:
        name = "stub"

        def generate(self, query: str, sources: list[object]) -> GeneratedAnswer:
            # Emits a valid [1] and a fabricated [9]; the latter must be removed.
            return GeneratedAnswer(
                text="A grounded claim [1] and a bogus one [9].", used_markers=[1, 9]
            )

    engine = QueryEngine(settings, store=store, generator=_Stub())  # type: ignore[arg-type]
    answer = engine.answer("transformers attention")
    assert answer.status is AnswerStatus.ANSWERED
    assert "[9]" not in answer.text
    assert "[1]" in answer.text
    assert all(c.number != 9 for c in answer.citations)


def test_unattributed_answer_becomes_refusal(settings: Settings, store: QdrantStore) -> None:
    settings.retrieval.refusal_min_score = 0.05
    ingest_corpus(settings, store=store)

    class _NoCite:
        name = "nocite"

        def generate(self, query: str, sources: list[object]) -> GeneratedAnswer:
            return GeneratedAnswer(text="A confident but uncited claim.", used_markers=[])

    engine = QueryEngine(settings, store=store, generator=_NoCite())  # type: ignore[arg-type]
    answer = engine.answer("transformers attention")
    assert answer.status is AnswerStatus.REFUSED


def test_generation_error_surfaces_as_error(settings: Settings, store: QdrantStore) -> None:
    settings.retrieval.refusal_min_score = 0.05
    ingest_corpus(settings, store=store)

    class _Boom:
        name = "boom"

        def generate(self, query: str, sources: list[object]) -> GeneratedAnswer:
            raise GenerationError("backend down")

    engine = QueryEngine(settings, store=store, generator=_Boom())  # type: ignore[arg-type]
    answer = engine.answer("transformers attention")
    assert answer.status is AnswerStatus.ERROR
    assert answer.error is not None


def test_recent_filter_uses_manifest_tags(settings: Settings, store: QdrantStore) -> None:
    # The engine derives its tag vocabulary from the manifest (corpus as a variable).
    engine = _ingested_engine(settings, store)
    a = analyze_query("recent prompting work", known_tags=engine._known_tags)
    assert "prompting" in a.tags
