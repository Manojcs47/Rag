"""Unit tests for agent tool helpers (M3)."""

from __future__ import annotations

from datetime import date

from research_navigator.agents.tools import (
    corpus_metadata_lookup,
    date_math,
    rank_papers_for_reading_list,
)
from research_navigator.ingest.manifest import DocumentMeta


def _doc(
    doc_id: str,
    *,
    tags: list[str] | None = None,
    year: int = 2024,
    is_foundational: bool = False,
    content_type: str = "arxiv_paper",
) -> DocumentMeta:
    return DocumentMeta(
        doc_id=doc_id,
        content_type=content_type,
        title=f"Title of {doc_id}",
        authors=["A"],
        year=year,
        primary_category="cs.CL",
        tags=tags or [],
        is_foundational=is_foundational,
        source_url=f"https://arxiv.org/abs/{doc_id}",
        local_path=f"documents/arxiv/{doc_id}.pdf",
    )


DOCS = [
    _doc("d1", tags=["RAG", "retrieval"], year=2024, is_foundational=False),
    _doc("d2", tags=["RAG"], year=2020, is_foundational=True),
    _doc("d3", tags=["agents"], year=2024),
    _doc("d4", tags=["transformers"], year=2017, is_foundational=True),
]


# ---- corpus_metadata_lookup ----------------------------------------------


def test_lookup_filters_by_tag_or_semantics() -> None:
    matches = corpus_metadata_lookup(DOCS, tags=["RAG"])
    ids = {m.doc_id for m in matches}
    assert ids == {"d1", "d2"}


def test_lookup_filters_by_foundational() -> None:
    matches = corpus_metadata_lookup(DOCS, is_foundational=True)
    assert {m.doc_id for m in matches} == {"d2", "d4"}


def test_lookup_filters_by_year_range() -> None:
    matches = corpus_metadata_lookup(DOCS, year_min=2024)
    assert {m.doc_id for m in matches} == {"d1", "d3"}


def test_lookup_by_doc_id_returns_single() -> None:
    matches = corpus_metadata_lookup(DOCS, doc_id="d4")
    assert len(matches) == 1
    assert matches[0].doc_id == "d4"


def test_lookup_combines_constraints() -> None:
    matches = corpus_metadata_lookup(DOCS, tags=["RAG"], is_foundational=True)
    assert {m.doc_id for m in matches} == {"d2"}


def test_lookup_empty_when_nothing_matches() -> None:
    assert corpus_metadata_lookup(DOCS, tags=["nonexistent_tag"]) == []


# ---- date_math -----------------------------------------------------------


def test_date_math_12_months_back() -> None:
    rng = date_math(months_back=12, today=date(2025, 6, 1))
    assert rng.year_min == 2024


def test_date_math_18_months_back() -> None:
    rng = date_math(months_back=18, today=date(2025, 6, 1))
    assert rng.year_min == 2023


def test_date_math_default_uses_today() -> None:
    rng = date_math(months_back=0, today=date(2026, 3, 1))
    assert rng.year_min == 2026


# ---- rank_papers_for_reading_list ----------------------------------------


def test_rank_foundational_first_when_requested() -> None:
    matches = corpus_metadata_lookup(DOCS, tags=["RAG"])
    ranked = rank_papers_for_reading_list(matches, foundational_first=True, topic_tags=["RAG"])
    assert ranked[0].is_foundational is True  # d2 should win


def test_rank_by_tag_match_count() -> None:
    # d1 matches 2 tags (RAG, retrieval), d2 matches 1.
    matches = corpus_metadata_lookup(DOCS, tags=["RAG", "retrieval"])
    ranked = rank_papers_for_reading_list(matches, topic_tags=["RAG", "retrieval"])
    assert ranked[0].doc_id == "d1"


def test_rank_by_year_when_other_signals_tied() -> None:
    matches = corpus_metadata_lookup(DOCS, tags=["RAG"])
    ranked = rank_papers_for_reading_list(matches, topic_tags=["RAG"])
    # Without foundational_first, d1 (2024) should outrank d2 (2020).
    assert ranked[0].year >= ranked[-1].year
