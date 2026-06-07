"""Tests for golden-set loading and the shipped golden_set.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_navigator.agents.state import AgentRoute
from research_navigator.eval.golden_set import GoldenItem, load_golden_set


def test_golden_set_loads_and_validates(tmp_path: Path) -> None:
    p = tmp_path / "g.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"id": "q1", "query": "What is RAG?", "route": "concept_explanation",'
                ' "expected_doc_ids": ["arxiv-2005.11401"]}',
                '{"id": "q2", "query": "Weather?", "route": "out_of_scope",'
                ' "expected_doc_ids": [], "is_out_of_corpus": true}',
            ]
        ),
        encoding="utf-8",
    )
    items = load_golden_set(p)
    assert len(items) == 2
    assert items[0].route is AgentRoute.CONCEPT_EXPLANATION
    assert items[1].expected_refusal is True


def test_golden_set_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    p = tmp_path / "g.jsonl"
    p.write_text(
        '// header comment\n\n{"id": "q1", "query": "x", "route": "concept_explanation"}\n\n',
        encoding="utf-8",
    )
    assert len(load_golden_set(p)) == 1


def test_golden_set_bad_line_raises_with_line_number(tmp_path: Path) -> None:
    p = tmp_path / "g.jsonl"
    p.write_text('{"id": "q1", "query": "x"}\n', encoding="utf-8")  # missing route
    with pytest.raises(ValueError, match=":1:"):
        load_golden_set(p)


def test_shipped_golden_set_validates() -> None:
    """The committed eval/golden_set.jsonl must parse and cover all six routes."""
    p = Path("eval/golden_set.jsonl")
    if not p.is_file():
        pytest.skip("eval/golden_set.jsonl not present in this checkout")
    items = load_golden_set(p)
    assert len(items) >= 40, f"golden set should have >= 40 entries, got {len(items)}"
    routes_seen = {item.route for item in items}
    for route in AgentRoute:
        assert route in routes_seen, f"golden set missing the {route.value} route"
    # At least one out-of-corpus, to exercise refusal_correctness.
    assert any(item.is_out_of_corpus for item in items)


def test_golden_item_round_trips() -> None:
    item = GoldenItem(
        id="q",
        query="x",
        route=AgentRoute.CONCEPT_EXPLANATION,
        expected_doc_ids=["d1"],
    )
    again = GoldenItem.model_validate(json.loads(item.model_dump_json()))
    assert again == item
