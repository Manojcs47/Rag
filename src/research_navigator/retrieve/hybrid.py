"""Hybrid retrieval: dense + sparse, fused with RRF inside Qdrant (M2, adr-0004).

A single :meth:`HybridRetriever.retrieve` call issues two Qdrant requests:

1. A fused ``query_points`` with two prefetch branches — dense (cosine) and sparse
   (BM25 via Qdrant's IDF modifier) — combined by Reciprocal Rank Fusion. The
   metadata filter is attached to *each* branch, so filtering happens server-side.
2. A dense-only ``query_points`` (same filter) whose top cosine similarity is the
   **confidence signal** for the refusal gate. The fused RRF score is rank-based and
   not comparable to a similarity threshold (adr-0007), so it must not be used here.

Results are returned as typed :class:`RetrievedChunk` objects carrying everything
generation and citation rendering need; no raw Qdrant payloads escape this module.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field
from qdrant_client import models

from research_navigator.config import Settings
from research_navigator.ingest.embed import DenseEmbedder, SparseEmbedder
from research_navigator.ingest.qdrant_store import QdrantStore
from research_navigator.logging import get_logger

log = get_logger(__name__)

# Payload keys pulled back from Qdrant for each hit (document + per-chunk metadata).
_PAYLOAD_FIELDS = [
    "doc_id",
    "content_type",
    "title",
    "authors",
    "year",
    "month",
    "primary_category",
    "tags",
    "is_foundational",
    "source_url",
    "section_title",
    "section_path",
    "section_index",
    "chunk_index",
    "kind",
    "text",
]


@runtime_checkable
class Retriever(Protocol):
    """Common interface implemented by every retriever (M2 hybrid, M4 ablations).

    Both :class:`HybridRetriever` and the ``DenseOnlyRetriever`` defined in the
    eval package satisfy this Protocol, so :class:`QueryEngine` can be parameterised
    with either at construction time.
    """

    def retrieve(
        self,
        query: str,
        *,
        query_filter: Any = None,
        candidate_k: int | None = None,
        prefetch_limit: int | None = None,
    ) -> RetrievalResult: ...


class RetrievedChunk(BaseModel):
    """One scored chunk with the metadata needed for generation + citation."""

    # Document-level (carried from the manifest via the chunk payload).
    doc_id: str
    content_type: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int
    month: int | None = None
    primary_category: str = ""
    tags: list[str] = Field(default_factory=list)
    is_foundational: bool = False
    source_url: str = ""
    # Chunk-level.
    section_title: str = ""
    section_path: list[str] = Field(default_factory=list)
    section_index: int = -1
    chunk_index: int = -1
    kind: str = "body"
    text: str = ""
    # Scores.
    score: float = 0.0  # fused RRF score (ranking)
    dense_score: float | None = None  # cosine similarity (confidence), when known

    @classmethod
    def from_point(cls, point: models.ScoredPoint) -> RetrievedChunk:
        """Build a chunk from a Qdrant scored point (isolates the untyped payload)."""
        payload: dict[str, Any] = point.payload or {}
        return cls(
            doc_id=str(payload.get("doc_id", "")),
            content_type=str(payload.get("content_type", "")),
            title=str(payload.get("title", "")),
            authors=list(payload.get("authors") or []),
            year=int(payload.get("year", 0)),
            month=payload.get("month"),
            primary_category=str(payload.get("primary_category", "")),
            tags=list(payload.get("tags") or []),
            is_foundational=bool(payload.get("is_foundational", False)),
            source_url=str(payload.get("source_url", "")),
            section_title=str(payload.get("section_title", "")),
            section_path=list(payload.get("section_path") or []),
            section_index=int(payload.get("section_index", -1)),
            chunk_index=int(payload.get("chunk_index", -1)),
            kind=str(payload.get("kind", "body")),
            text=str(payload.get("text", "")),
            score=float(point.score),
        )


class RetrievalResult(BaseModel):
    """The ranked chunks for a query plus the confidence signal for refusal."""

    chunks: list[RetrievedChunk] = Field(default_factory=list)
    dense_confidence: float = 0.0  # max dense cosine among filtered candidates

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class HybridRetriever:
    """Dense + sparse retrieval fused with RRF, filtered inside Qdrant."""

    def __init__(
        self,
        settings: Settings,
        store: QdrantStore,
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
    ) -> None:
        self._s = settings
        self._store = store
        self._dense = dense
        self._sparse = sparse

    def retrieve(
        self,
        query: str,
        *,
        query_filter: models.Filter | None = None,
        candidate_k: int | None = None,
        prefetch_limit: int | None = None,
    ) -> RetrievalResult:
        """Retrieve fused candidates for ``query`` under ``query_filter``."""
        candidate_k = candidate_k or self._s.retrieval.candidate_k
        prefetch_limit = prefetch_limit or self._s.retrieval.prefetch_limit
        collection = self._s.qdrant_collection

        dense_vec = self._dense.embed_query(query)
        sparse_vec = self._sparse.embed_query(query)

        fused = self._store.client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using=self._s.dense_vector_name,
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                    using=self._s.sparse_vector_name,
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=candidate_k,
            with_payload=_PAYLOAD_FIELDS,
        )
        chunks = [RetrievedChunk.from_point(p) for p in fused.points]

        # Separate dense-only pass for a similarity-calibrated confidence signal.
        confidence = self._dense_confidence(dense_vec, query_filter)
        log.info(
            "retrieve_done",
            results=len(chunks),
            dense_confidence=round(confidence, 4),
            filtered=query_filter is not None,
        )
        return RetrievalResult(chunks=chunks, dense_confidence=confidence)

    def _dense_confidence(
        self, dense_vec: list[float], query_filter: models.Filter | None
    ) -> float:
        """Top dense cosine similarity among filtered candidates (0.0 if none)."""
        res = self._store.client.query_points(
            collection_name=self._s.qdrant_collection,
            query=dense_vec,
            using=self._s.dense_vector_name,
            query_filter=query_filter,
            limit=1,
            with_payload=False,
        )
        return float(res.points[0].score) if res.points else 0.0
