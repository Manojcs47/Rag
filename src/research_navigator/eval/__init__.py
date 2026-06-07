"""Evaluation harness (M4).

A reproducible, automated measurement of the system's retrieval precision/recall,
citation faithfulness, refusal correctness, latency, and approximate token cost,
across multiple retrieval configurations. The headline output is a one-page
Markdown report plus a JSON dump of every per-query verdict for drill-down.
"""

from __future__ import annotations

from research_navigator.eval.golden_set import GoldenItem, load_golden_set
from research_navigator.eval.metrics import latency_percentiles, precision_at_k, recall_at_k
from research_navigator.eval.report import write_report
from research_navigator.eval.retrievers import DenseOnlyRetriever
from research_navigator.eval.runner import EvalConfig, EvalReport, run_evaluation

__all__ = [
    "DenseOnlyRetriever",
    "EvalConfig",
    "EvalReport",
    "GoldenItem",
    "latency_percentiles",
    "load_golden_set",
    "precision_at_k",
    "recall_at_k",
    "run_evaluation",
    "write_report",
]
