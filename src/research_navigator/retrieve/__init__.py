"""Retrieval: query understanding, metadata-filter inference, and hybrid search.

This package turns a learner's free-text question into (a) a structured
:class:`~research_navigator.retrieve.query.QueryAnalysis` of inferable metadata
filters and (b) a ranked list of grounded chunks via dense + sparse hybrid search
fused with RRF inside Qdrant (filters applied server-side, never post-hoc).
"""

from __future__ import annotations

from research_navigator.retrieve.hybrid import (
    HybridRetriever,
    RetrievalResult,
    RetrievedChunk,
)
from research_navigator.retrieve.query import QueryAnalysis, analyze_query

__all__ = [
    "HybridRetriever",
    "QueryAnalysis",
    "RetrievalResult",
    "RetrievedChunk",
    "analyze_query",
]
