"""M2 query pipeline: retrieve -> refuse-or-generate -> validate citations.

:class:`QueryEngine` wires the retriever (M2 retrieval) and a generator backend
into the end-to-end contract:

  1. Understand the query and infer Qdrant metadata filters.
  2. Retrieve hybrid candidates (filters applied server-side). If a tag filter
     empties the result, relax it once and retry (logged fallback, never silent).
  3. **Refuse** when the dense-cosine confidence is below the tuned threshold, or
     nothing was retrieved.
  4. Build deduplicated, numbered citations; generate an answer whose every factual
     sentence carries a ``[n]`` marker.
  5. **Validate** markers against the real citation set — out-of-range markers are
     stripped, and an answer that ends up with *no* valid attribution is converted
     to a refusal (an unattributed claim is not allowed to stand).

The engine is constructed once (it may load an embedding model) and reused.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from research_navigator.config import Settings, get_settings
from research_navigator.generate.citations import (
    Citation,
    build_citations,
    render_citation_block,
)
from research_navigator.generate.generator import (
    GenerationError,
    Generator,
    build_generator,
)
from research_navigator.generate.prompt import REFUSAL_TEXT
from research_navigator.ingest.embed import DenseEmbedder, SparseEmbedder, build_embedders
from research_navigator.ingest.manifest import load_manifest
from research_navigator.ingest.qdrant_store import QdrantStore
from research_navigator.logging import get_logger
from research_navigator.retrieve.filters import build_filter
from research_navigator.retrieve.hybrid import HybridRetriever, RetrievedChunk
from research_navigator.retrieve.query import QueryAnalysis, analyze_query

log = get_logger(__name__)

_MARKER = re.compile(r"\[(\d+)\]")


class AnswerStatus(StrEnum):
    """Outcome of a query."""

    ANSWERED = "answered"
    REFUSED = "refused"
    ERROR = "error"


class Answer(BaseModel):
    """A complete response: answer text, citations, and the signals behind it."""

    status: AnswerStatus
    query: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    num_retrieved: int = 0
    analysis: QueryAnalysis | None = None
    error: str | None = None

    @property
    def refused(self) -> bool:
        return self.status is AnswerStatus.REFUSED

    def render(self) -> str:
        """Markdown rendering: answer body followed by the sources block."""
        if self.status is not AnswerStatus.ANSWERED:
            return self.text
        block = render_citation_block(self.citations)
        return f"{self.text}\n\n{block}" if block else self.text


class QueryEngine:
    """Answers learner questions with grounded citations, or refuses."""

    def __init__(
        self,
        settings: Settings,
        *,
        store: QdrantStore | None = None,
        dense: DenseEmbedder | None = None,
        sparse: SparseEmbedder | None = None,
        generator: Generator | None = None,
        known_tags: frozenset[str] | None = None,
    ) -> None:
        self._s = settings
        self._store = store or QdrantStore(settings)
        if dense is None or sparse is None:
            built_dense, built_sparse = build_embedders(settings)
            dense = dense or built_dense
            sparse = sparse or built_sparse
        self._retriever = HybridRetriever(settings, self._store, dense, sparse)
        self._generator = generator or build_generator(settings)
        self._known_tags = known_tags if known_tags is not None else self._load_tags()

    # ---- public properties used by M3 nodes -------------------------------

    @property
    def settings(self) -> Settings:
        return self._s

    @property
    def retriever(self) -> HybridRetriever:
        return self._retriever

    @property
    def generator(self) -> Generator:
        return self._generator

    def analyze(self, query: str) -> QueryAnalysis:
        """Public access to the configured filter-inference step."""
        return analyze_query(
            query,
            known_tags=self._known_tags,
            recent_year_floor=self._s.retrieval.recent_year_floor,
        )

    def _load_tags(self) -> frozenset[str] | None:
        """Derive the tag vocabulary from the manifest (corpus as a variable)."""
        try:
            manifest = load_manifest(self._s.manifest_path)
        except Exception as exc:
            log.warning("tag_vocab_from_manifest_failed", error=str(exc))
            return None
        tags = {t for doc in manifest.documents for t in doc.tags}
        return frozenset(tags) if tags else None

    # ---- M2: the original top-level answer() flow -------------------------

    def answer(self, query: str) -> Answer:
        """Run the full M2 pipeline for one question."""
        analysis = self.analyze(query)
        result = self._retrieve_with_relaxation(query, analysis)
        return self._finalize(query, analysis, result)

    # ---- M3 hooks ---------------------------------------------------------

    def answer_with_filter(self, query: str, query_filter: Any) -> Answer:
        """Answer using a caller-supplied Qdrant filter (skips analysis-derived filter).

        Used by PaperDeepDive (force a single doc_id) and any other node that
        needs to express a constraint outside the inferable-filter grammar.
        """
        analysis = self.analyze(query)
        result = self._retriever.retrieve(query, query_filter=query_filter)
        return self._finalize(query, analysis, result)

    def answer_with_analysis(self, query: str, analysis: QueryAnalysis) -> Answer:
        """Answer using a caller-supplied :class:`QueryAnalysis` (e.g. forced year_min).

        Used by RecentDevelopments to compose its date-math-driven recency floor
        with whatever else the query inferred (tags, content_type).
        """
        result = self._retrieve_with_relaxation(query, analysis)
        return self._finalize(query, analysis, result)

    def retrieve_with_filter(self, query: str, query_filter: Any):  # type: ignore[no-untyped-def]
        """Retrieval-only with a caller-supplied filter. Used by CompareApproaches."""
        return self._retriever.retrieve(query, query_filter=query_filter)

    def answer_from_chunks(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        confidence: float,
    ) -> Answer:
        """Answer from a pre-combined chunk list (e.g. union of multiple retrievals)."""
        from research_navigator.retrieve.hybrid import RetrievalResult

        analysis = self.analyze(query)
        result = RetrievalResult(chunks=chunks, dense_confidence=confidence)
        return self._finalize(query, analysis, result)

    # ---- internal: shared retrieval + finalisation ------------------------

    def _retrieve_with_relaxation(self, query: str, analysis: QueryAnalysis):  # type: ignore[no-untyped-def]
        """Retrieve under ``analysis``'s filter; relax tags once if it empties."""
        result = self._retriever.retrieve(query, query_filter=build_filter(analysis))
        if result.is_empty and analysis.tags:
            log.info("retrieval_filter_relaxed", reason="empty_with_tags")
            result = self._retriever.retrieve(
                query, query_filter=build_filter(analysis, include_tags=False)
            )
        return result

    def _finalize(self, query: str, analysis: QueryAnalysis, result) -> Answer:  # type: ignore[no-untyped-def]
        """Refusal gate → citations → generate → validate. The M2 contract."""
        if result.is_empty or result.dense_confidence < self._s.retrieval.refusal_min_score:
            log.info(
                "answer_refused",
                reason="empty" if result.is_empty else "low_confidence",
                confidence=round(result.dense_confidence, 4),
            )
            return Answer(
                status=AnswerStatus.REFUSED,
                query=query,
                text=REFUSAL_TEXT,
                confidence=result.dense_confidence,
                num_retrieved=len(result.chunks),
                analysis=analysis,
            )

        citations, _ = build_citations(result.chunks[: self._s.retrieval.top_k])
        sources = self._sources_for(citations, result.chunks)

        try:
            generated = self._generator.generate(query, sources)
        except GenerationError as exc:
            log.error("generation_failed", error=str(exc))
            return Answer(
                status=AnswerStatus.ERROR,
                query=query,
                text="",
                confidence=result.dense_confidence,
                num_retrieved=len(result.chunks),
                analysis=analysis,
                error=str(exc),
            )

        valid = {c.number for c in citations}
        cleaned = _strip_invalid_markers(generated.text, valid)
        referenced = {int(m.group(1)) for m in _MARKER.finditer(cleaned)}
        if not referenced:
            log.warning("answer_unattributed_refused", confidence=result.dense_confidence)
            return Answer(
                status=AnswerStatus.REFUSED,
                query=query,
                text=REFUSAL_TEXT,
                confidence=result.dense_confidence,
                num_retrieved=len(result.chunks),
                analysis=analysis,
            )

        used_citations = [c for c in citations if c.number in referenced]
        log.info("answer_done", citations=referenced, confidence=round(result.dense_confidence, 4))
        return Answer(
            status=AnswerStatus.ANSWERED,
            query=query,
            text=cleaned.strip(),
            citations=used_citations,
            confidence=result.dense_confidence,
            num_retrieved=len(result.chunks),
            analysis=analysis,
        )

    @staticmethod
    def _sources_for(
        citations: list[Citation], chunks: list[RetrievedChunk]
    ) -> list[tuple[Citation, RetrievedChunk]]:
        """Pair each citation with its most relevant (first-seen, best) chunk."""
        first_by_doc: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            first_by_doc.setdefault(chunk.doc_id, chunk)
        return [(c, first_by_doc[c.doc_id]) for c in citations if c.doc_id in first_by_doc]


def _strip_invalid_markers(text: str, valid: set[int]) -> str:
    """Remove ``[n]`` markers whose number is not a real citation; tidy spacing."""

    def repl(m: re.Match[str]) -> str:
        return m.group(0) if int(m.group(1)) in valid else ""

    cleaned = _MARKER.sub(repl, text)
    return re.sub(r"[ \t]+([.!?,;])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned))


def answer_query(query: str, settings: Settings | None = None) -> Answer:
    """Convenience one-shot: build an engine and answer a single query.

    Prefer constructing a :class:`QueryEngine` once and reusing it in hot paths
    (it may load an embedding model); this helper exists for tests and scripts.
    """
    settings = settings or get_settings()
    return QueryEngine(settings).answer(query)
