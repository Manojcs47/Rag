"""Unit tests for translating query analysis into Qdrant filters (M2)."""

from __future__ import annotations

from qdrant_client import models

from research_navigator.retrieve.filters import build_filter
from research_navigator.retrieve.query import QueryAnalysis


def _keys(flt: models.Filter | None) -> list[str]:
    assert flt is not None
    return [c.key for c in flt.must]  # type: ignore[union-attr]


def test_no_constraints_returns_none() -> None:
    assert build_filter(QueryAnalysis(raw_query="x")) is None


def test_year_range_condition() -> None:
    flt = build_filter(QueryAnalysis(raw_query="x", year_min=2023, year_max=2025))
    assert _keys(flt) == ["year"]
    cond = flt.must[0]  # type: ignore[union-attr,index]
    assert cond.range.gte == 2023  # type: ignore[union-attr]
    assert cond.range.lte == 2025  # type: ignore[union-attr]


def test_tags_use_match_any_for_or_semantics() -> None:
    flt = build_filter(QueryAnalysis(raw_query="x", tags=["RAG", "agents"]))
    cond = flt.must[0]  # type: ignore[union-attr,index]
    assert cond.key == "tags"
    assert isinstance(cond.match, models.MatchAny)
    assert set(cond.match.any) == {"RAG", "agents"}


def test_include_tags_false_drops_tag_condition() -> None:
    analysis = QueryAnalysis(raw_query="x", tags=["RAG"], year_min=2024)
    full = build_filter(analysis, include_tags=True)
    relaxed = build_filter(analysis, include_tags=False)
    assert "tags" in _keys(full)
    assert _keys(relaxed) == ["year"]


def test_content_type_and_foundational_conditions() -> None:
    flt = build_filter(
        QueryAnalysis(raw_query="x", content_types=["survey_blog"], is_foundational=True)
    )
    keys = _keys(flt)
    assert "content_type" in keys
    assert "is_foundational" in keys
