"""Explicit, serialisable state for the LangGraph agent (M3).

The brief requires state to be "explicit and serialisable". We model it with
pydantic so every field is typed, validated, and round-trips via JSON for any
future checkpoint/replay use case (M4 / debugging).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from research_navigator.generate.pipeline import Answer


class AgentRoute(StrEnum):
    """The six sub-capabilities mandated by the brief, plus the out-of-scope path."""

    CONCEPT_EXPLANATION = "concept_explanation"
    PAPER_DEEP_DIVE = "paper_deep_dive"
    COMPARE_APPROACHES = "compare_approaches"
    RECENT_DEVELOPMENTS = "recent_developments"
    FIND_PAPERS = "find_papers"
    OUT_OF_SCOPE = "out_of_scope"


class RoutingHints(BaseModel):
    """Structured hints the router extracts and downstream nodes consume.

    Keeping these on the state (rather than re-parsing inside each node) makes
    the run inspectable and avoids duplicated classification work.
    """

    paper_doc_id: str | None = None
    """PaperDeepDive: doc_id of the paper named or implied by the query."""

    compare_targets: list[str] = Field(default_factory=list)
    """CompareApproaches: 2+ named methods/papers to compare."""

    foundational_only: bool = False
    """FindPapers: prefer foundational papers (e.g. 'seminal papers on X')."""

    recent_only: bool = False
    """FindPapers/RecentDevelopments: restrict to the recent window."""


class ToolInvocation(BaseModel):
    """One tool call made inside a node, recorded for traceability."""

    name: str
    args: dict[str, object] = Field(default_factory=dict)
    result_summary: str = ""


class AgentState(BaseModel):
    """The full agent state. JSON-serialisable; passed by LangGraph between nodes."""

    query: str
    route: AgentRoute | None = None
    hints: RoutingHints = Field(default_factory=RoutingHints)
    answer: Answer | None = None
    tool_log: list[ToolInvocation] = Field(default_factory=list)
