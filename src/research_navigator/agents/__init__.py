"""LangGraph agentic layer (M3): route queries into specialised sub-capabilities.

The router classifies a learner's question into one of six routes — concept
explanation, paper deep dive, approach comparison, recent developments, paper
finding, or out-of-scope — and a single sub-capability node produces the answer.
Each node consumes the shared :class:`AgentState`, which is explicit, pydantic-
typed, and JSON-serialisable for checkpointing.
"""

from __future__ import annotations

from research_navigator.agents.graph import (
    build_agent_graph,
    render_graph_mermaid,
    save_graph_visualization,
)
from research_navigator.agents.router import classify_query
from research_navigator.agents.state import AgentRoute, AgentState, RoutingHints, ToolInvocation

__all__ = [
    "AgentRoute",
    "AgentState",
    "RoutingHints",
    "ToolInvocation",
    "build_agent_graph",
    "classify_query",
    "render_graph_mermaid",
    "save_graph_visualization",
]
