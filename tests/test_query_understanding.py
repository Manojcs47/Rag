"""Unit tests for query understanding / metadata-filter inference (M2)."""

from __future__ import annotations

from research_navigator.retrieve.query import (
    DEFAULT_TAG_VOCABULARY,
    analyze_query,
)


def test_recent_maps_to_year_floor() -> None:
    a = analyze_query("recent work on RAG", recent_year_floor=2024)
    assert a.wants_recent is True
    assert a.year_min == 2024
    assert a.year_max is None
    assert "RAG" in a.tags


def test_latest_is_recency_signal() -> None:
    a = analyze_query("latest open-weight models", recent_year_floor=2024)
    assert a.wants_recent is True
    assert a.year_min == 2024
    assert "open_models" in a.tags


def test_explicit_since_and_before_bounds() -> None:
    a = analyze_query("alignment papers since 2023 and before 2025")
    assert a.year_min == 2023
    assert a.year_max == 2025
    assert "alignment" in a.tags


def test_in_year_sets_exact_bounds() -> None:
    a = analyze_query("transformers published in 2017")
    assert a.year_min == 2017
    assert a.year_max == 2017


def test_bare_year_is_lower_bound() -> None:
    a = analyze_query("benchmarks from 2024 onwards-ish")
    assert a.year_min == 2024


def test_tag_aliases_resolve_to_vocabulary() -> None:
    a = analyze_query("compare chain-of-thought with reinforcement learning reasoning")
    assert "chain_of_thought" in a.tags
    assert "RL" in a.tags
    assert "reasoning" in a.tags


def test_content_type_hint_inferred() -> None:
    a = analyze_query("which survey covers autonomous agents?")
    assert "survey_blog" in a.content_types
    assert "agents" in a.tags


def test_foundational_hint_inferred() -> None:
    a = analyze_query("what is the seminal transformer paper?")
    assert a.is_foundational is True


def test_no_signals_means_no_filters() -> None:
    a = analyze_query("tell me about embeddings")
    assert not a.has_filters
    assert a.tags == []


def test_unknown_tags_are_ignored_when_vocab_restricted() -> None:
    a = analyze_query("RAG and agents", known_tags=frozenset({"agents"}))
    assert a.tags == ["agents"]  # RAG not in the restricted vocabulary


def test_default_vocabulary_is_nonempty() -> None:
    assert "LLM" in DEFAULT_TAG_VOCABULARY
    assert "RAG" in DEFAULT_TAG_VOCABULARY
