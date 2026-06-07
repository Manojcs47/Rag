"""Integration test for the M4 runner using the fixture corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_navigator.config import Settings
from research_navigator.eval.report import write_report
from research_navigator.eval.runner import EvalConfig, run_evaluation
from research_navigator.ingest.pipeline import ingest_corpus
from research_navigator.ingest.qdrant_store import QdrantStore

pytestmark = pytest.mark.integration

_TINY_GOLDEN = (
    '{"id": "t1", "query": "What is prompt engineering?",'
    ' "route": "concept_explanation",'
    ' "expected_doc_ids": ["lillog-prompt-eng-2023-03"]}\n'
    '{"id": "t2", "query": "Weather in Hyderabad?",'
    ' "route": "out_of_scope", "expected_doc_ids": [], "is_out_of_corpus": true}\n'
)


@pytest.fixture
def tiny_golden(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.jsonl"
    p.write_text(_TINY_GOLDEN, encoding="utf-8")
    return p


def test_runner_produces_report(
    settings: Settings,
    store: QdrantStore,
    tiny_golden: Path,
    tmp_path: Path,
) -> None:
    # Floor the refusal gate — the offline embedder has no real semantic geometry.
    settings.retrieval.refusal_min_score = 0.0
    ingest_corpus(settings, store=store)

    report = run_evaluation(
        tiny_golden,
        settings=settings,
        configs=[
            EvalConfig(name="hybrid_with_filters"),
            EvalConfig(name="dense_only_with_filters", dense_only=True),
        ],
        store=store,
    )

    assert report.n_questions == 2
    assert len(report.configs) == 2
    assert {c.name for c in report.configs} == {
        "hybrid_with_filters",
        "dense_only_with_filters",
    }
    # The OOC question must register as a refusal-correctness sample.
    assert all(c.n_out_of_corpus == 1 for c in report.configs)


def test_report_writes_both_files(
    settings: Settings,
    store: QdrantStore,
    tiny_golden: Path,
    tmp_path: Path,
) -> None:
    settings.retrieval.refusal_min_score = 0.0
    ingest_corpus(settings, store=store)

    report = run_evaluation(
        tiny_golden,
        settings=settings,
        configs=[EvalConfig(name="hybrid_with_filters")],
        store=store,
    )
    out_dir = tmp_path / "report_out"
    json_path, md_path = write_report(report, out_dir)
    assert json_path.is_file() and md_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["n_questions"] == 2
    assert "# AI Research Navigator — Evaluation Report" in md_path.read_text(encoding="utf-8")
