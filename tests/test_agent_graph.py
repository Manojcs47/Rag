"""Integration test for the LangGraph agent (M3).

Uses the existing in-memory Qdrant + offline-embedder fixture from conftest.
Ingests the fixture corpus once and exercises the graph end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_navigator.agents import build_agent_graph, render_graph_mermaid
from research_navigator.agents.state import AgentRoute, AgentState
from research_navigator.config import Settings
from research_navigator.generate.pipeline import QueryEngine
from research_navigator.ingest.pipeline import ingest_corpus
from research_navigator.ingest.qdrant_store import QdrantStore

pytestmark = pytest.mark.integration


@pytest.fixture
def ingested(settings: Settings, store: QdrantStore) -> QdrantStore:
    """Ingest the fixture corpus once for the graph tests."""
    ingest_corpus(settings, store=store)
    return store


def _engine(settings: Settings, store: QdrantStore) -> QueryEngine:
    # Force a very low refusal floor — the offline hashing embedder has no
    # semantic geometry, so cosine values stay low even on perfect matches.
    settings.retrieval.refusal_min_score = 0.0
    return QueryEngine(settings, store=store)


def test_graph_renders_mermaid(
    settings: Settings, store: QdrantStore, ingested: QdrantStore
) -> None:
    graph = build_agent_graph(settings, engine=_engine(settings, store))
    mermaid = render_graph_mermaid(graph)
    # Must mention every node id.
    for route in AgentRoute:
        assert route.value in mermaid
    assert "router" in mermaid


def test_route_concept_explanation(
    settings: Settings, store: QdrantStore, ingested: QdrantStore
) -> None:
    graph = build_agent_graph(settings, engine=_engine(settings, store))
    final = AgentState.model_validate(graph.invoke(AgentState(query="What is a neural network?")))
    assert final.route is AgentRoute.CONCEPT_EXPLANATION
    assert final.answer is not None


def test_route_find_papers(settings: Settings, store: QdrantStore, ingested: QdrantStore) -> None:
    graph = build_agent_graph(settings, engine=_engine(settings, store))
    final = AgentState.model_validate(
        graph.invoke(AgentState(query="Recommend papers on prompting"))
    )
    assert final.route is AgentRoute.FIND_PAPERS
    assert final.tool_log, "find_papers must record at least one tool call"
    # Output is metadata-grounded; status should be ANSWERED (no LLM gate).
    assert final.answer is not None and final.answer.status.value == "answered"


def test_route_out_of_scope(settings: Settings, store: QdrantStore, ingested: QdrantStore) -> None:
    graph = build_agent_graph(settings, engine=_engine(settings, store))
    final = AgentState.model_validate(
        graph.invoke(AgentState(query="What's the weather in Mumbai today?"))
    )
    assert final.route is AgentRoute.OUT_OF_SCOPE
    assert final.answer is not None and final.answer.status.value == "refused"


def test_visualization_writes_file(
    tmp_path: Path, settings: Settings, store: QdrantStore, ingested: QdrantStore
) -> None:
    from research_navigator.agents import save_graph_visualization

    graph = build_agent_graph(settings, engine=_engine(settings, store))
    out = save_graph_visualization(tmp_path / "graph.md", graph)
    body = out.read_text(encoding="utf-8")
    assert "```mermaid" in body
    assert "router" in body
