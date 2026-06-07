"""Unit tests for eval metrics (M4)."""

from __future__ import annotations

from research_navigator.eval.metrics import (
    approx_tokens,
    latency_percentiles,
    precision_at_k,
    recall_at_k,
)


def test_precision_perfect() -> None:
    assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_precision_partial() -> None:
    assert precision_at_k(["a", "b", "c"], {"a"}, k=3) == 1 / 3


def test_precision_empty_retrieved() -> None:
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_precision_dedups_retrieved() -> None:
    # Duplicate doc_ids in input must be collapsed before scoring.
    assert precision_at_k(["a", "a", "b"], {"a"}, k=2) == 0.5


def test_recall_perfect() -> None:
    assert recall_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0


def test_recall_partial() -> None:
    assert recall_at_k(["a"], {"a", "b"}, k=5) == 0.5


def test_recall_empty_expected_is_vacuously_one() -> None:
    assert recall_at_k(["a"], set(), k=5) == 1.0


def test_recall_top_k_limit() -> None:
    # Only the first 2 retrieved count; "c" never gets considered.
    assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_latency_percentiles_empty() -> None:
    out = latency_percentiles([])
    assert out["p50"] == 0.0 and out["p95"] == 0.0


def test_latency_percentiles_single() -> None:
    out = latency_percentiles([42.0])
    assert out["p50"] == 42.0 and out["p95"] == 42.0


def test_latency_percentiles_basic() -> None:
    out = latency_percentiles([10.0, 20.0, 30.0, 40.0, 50.0])
    assert out["p50"] == 30.0
    assert out["max"] == 50.0


def test_approx_tokens_grows_with_text() -> None:
    short = approx_tokens("hello")
    longer = approx_tokens("hello there", "general kenobi")
    assert longer > short


def test_approx_tokens_min_one() -> None:
    assert approx_tokens("") == 1
