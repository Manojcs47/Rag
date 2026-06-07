"""Ablation retrievers for the eval harness (M4).

Currently exposes :class:`DenseOnlyRetriever`, which mirrors
:class:`HybridRetriever` minus the sparse branch. Used in the
``dense_only_with_filters`` configuration to quantify what sparse contributes.
"""

from __future__ import annotations

from typing import Any

from research_navigator.config import Settings
from research_navigator.ingest.embed import DenseEmbedder
from research_navigator.ingest.qdrant_store import QdrantStore
from research_navigator.logging import get_logger
from research_navigator.retrieve.hybrid import (
    _PAYLOAD_FIELDS,
    RetrievalResult,
    RetrievedChunk,
)

log = get_logger(__name__)


class DenseOnlyRetriever:
    """Dense-only retrieval (no sparse, no RRF). Satisfies the ``Retriever`` Protocol."""

    name = "dense_only"

    def __init__(self, settings: Settings, store: QdrantStore, dense: DenseEmbedder) -> None:
        self._s = settings
        self._store = store
        self._dense = dense

    def retrieve(
        self,
        query: str,
        *,
        query_filter: Any = None,
        candidate_k: int | None = None,
        prefetch_limit: int | None = None,
    ) -> RetrievalResult:
        """Retrieve dense-only candidates for ``query`` under ``query_filter``."""
        candidate_k = candidate_k or self._s.retrieval.candidate_k
        dense_vec = self._dense.embed_query(query)
        res = self._store.client.query_points(
            collection_name=self._s.qdrant_collection,
            query=dense_vec,
            using=self._s.dense_vector_name,
            query_filter=query_filter,
            limit=candidate_k,
            with_payload=_PAYLOAD_FIELDS,
        )
        chunks = [RetrievedChunk.from_point(p) for p in res.points]
        confidence = float(res.points[0].score) if res.points else 0.0
        log.info(
            "dense_only_retrieve_done",
            results=len(chunks),
            dense_confidence=round(confidence, 4),
            filtered=query_filter is not None,
        )
        return RetrievalResult(chunks=chunks, dense_confidence=confidence)
