"""The six agent nodes + Fallback (M3).

Each node receives the shared :class:`AgentState`, performs one focused step
(specialised retrieval, metadata ranking, recency filter, etc.), and writes its
result back as an :class:`Answer`. Nodes are wired into the graph by
``graph.py``; the helpers below take a :class:`NodeDeps` so the graph layer can
inject shared dependencies (a built :class:`QueryEngine`, the manifest) and
keep node functions unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import models

from research_navigator.agents.state import AgentState, ToolInvocation
from research_navigator.agents.tools import (
    CorpusMetadataMatch,
    corpus_metadata_lookup,
    date_math,
    rank_papers_for_reading_list,
)
from research_navigator.config import Settings
from research_navigator.generate.citations import Citation, format_authors
from research_navigator.generate.pipeline import Answer, AnswerStatus, QueryEngine
from research_navigator.generate.prompt import REFUSAL_TEXT
from research_navigator.ingest.manifest import DocumentMeta
from research_navigator.logging import get_logger
from research_navigator.retrieve.query import analyze_query

log = get_logger(__name__)

OUT_OF_SCOPE_TEXT = (
    "This question doesn't appear to be about AI/ML research, which is the scope "
    "of this corpus. Try asking about topics like LLMs, RAG, agents, alignment, "
    "fine-tuning, reasoning, or specific papers in the corpus."
)


@dataclass
class NodeDeps:
    """Shared dependencies injected into every node."""

    settings: Settings
    engine: QueryEngine
    documents: list[DocumentMeta]


# ---------- concept_explanation --------------------------------------------


def concept_explanation_node(state: AgentState, deps: NodeDeps) -> dict[str, Any]:
    """Synthesis-oriented explanation across multiple corpus sources (M2 directly)."""
    log.info("node_concept_explanation", query=state.query)
    return {"answer": deps.engine.answer(state.query)}


# ---------- paper_deep_dive ------------------------------------------------


def paper_deep_dive_node(state: AgentState, deps: NodeDeps) -> dict[str, Any]:
    """Retrieve preferentially from one specific paper's chunks."""
    doc_id = state.hints.paper_doc_id
    log.info("node_paper_deep_dive", query=state.query, doc_id=doc_id)

    if not doc_id:
        return {"answer": deps.engine.answer(state.query)}

    # TOOL CALL: confirm the paper exists in the manifest.
    matches = corpus_metadata_lookup(deps.documents, doc_id=doc_id)
    tool_log = [
        ToolInvocation(
            name="corpus_metadata_lookup",
            args={"doc_id": doc_id},
            result_summary=f"found {len(matches)} matching document(s)",
        )
    ]
    if not matches:
        return {
            "tool_log": tool_log,
            "answer": Answer(
                status=AnswerStatus.REFUSED,
                query=state.query,
                text=REFUSAL_TEXT,
            ),
        }

    # Build a doc_id-restricted filter and route through the M2 hook.
    doc_filter = models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )
    answer = deps.engine.answer_with_filter(state.query, doc_filter)
    return {"tool_log": tool_log, "answer": answer}


# ---------- compare_approaches ---------------------------------------------


def compare_approaches_node(state: AgentState, deps: NodeDeps) -> dict[str, Any]:
    """Retrieve from each named target, combine, produce a comparison answer."""
    targets = state.hints.compare_targets
    log.info("node_compare_approaches", query=state.query, targets=targets)

    if len(targets) < 2:
        return {"answer": deps.engine.answer(state.query)}

    tool_log: list[ToolInvocation] = []
    all_chunks = []
    confidences: list[float] = []

    for target in targets[:3]:
        # TOOL CALL: locate documents matching this target (tag, then title).
        tag_matches = corpus_metadata_lookup(
            deps.documents, tags=[target, target.replace(" ", "_")]
        )
        title_matches = [
            _doc_to_match(d) for d in deps.documents if target.lower() in d.title.lower()
        ]
        matches = tag_matches or title_matches
        tool_log.append(
            ToolInvocation(
                name="corpus_metadata_lookup",
                args={"target": target},
                result_summary=f"matched {len(matches)} document(s)",
            )
        )
        if not matches:
            continue

        target_doc_ids = [m.doc_id for m in matches[:2]]
        target_filter = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=target_doc_ids))]
        )
        result = deps.engine.retrieve_with_filter(f"{state.query} {target}", target_filter)
        # Top 4 per target keeps the combined source set manageable for citations.
        all_chunks.extend(result.chunks[:4])
        confidences.append(result.dense_confidence)

    if not all_chunks:
        return {"tool_log": tool_log, "answer": deps.engine.answer(state.query)}

    # Use the max confidence — even one strongly-matched target is enough to attempt
    # the comparison. The marker-validation step still degrades to a refusal if
    # neither side of the comparison produced anything attributable (ADR-0006).
    confidence = max(confidences) if confidences else 0.0
    answer = deps.engine.answer_from_chunks(state.query, all_chunks, confidence)
    return {"tool_log": tool_log, "answer": answer}


# ---------- recent_developments --------------------------------------------


def recent_developments_node(state: AgentState, deps: NodeDeps) -> dict[str, Any]:
    """Apply a recency floor and return a chronologically ordered digest."""
    log.info("node_recent_developments", query=state.query)

    # TOOL CALL: compute the recency floor from today.
    date_range = date_math(months_back=18)
    tool_log = [
        ToolInvocation(
            name="date_math",
            args={"months_back": 18},
            result_summary=f"year_min={date_range.year_min}",
        )
    ]

    # Compose with the inferable analysis (preserves tags / content-type hints).
    analysis = analyze_query(state.query, recent_year_floor=date_range.year_min)
    analysis.year_min = max(analysis.year_min or 0, date_range.year_min)
    analysis.wants_recent = True

    answer = deps.engine.answer_with_analysis(state.query, analysis)

    # Sort surfaced citations newest-first. Year is the primary key (manifest
    # ``month`` is nullable for some content types — Roadmap §0 finding #5).
    if answer.status is AnswerStatus.ANSWERED and answer.citations:
        answer.citations.sort(key=lambda c: c.year, reverse=True)

    return {"tool_log": tool_log, "answer": answer}


# ---------- find_papers ----------------------------------------------------


def find_papers_node(state: AgentState, deps: NodeDeps) -> dict[str, Any]:
    """Recommend a reading list via metadata ranking (no LLM generation).

    The brief explicitly says this route "relies on metadata filters
    (is_foundational, citation_count, year) rather than free-form generation".
    """
    log.info("node_find_papers", query=state.query)

    # Infer the topic from the query (tag vocabulary match).
    analysis = analyze_query(state.query)
    topic_tags = analysis.tags

    # TOOL CALL: gather candidate papers.
    matches = corpus_metadata_lookup(
        deps.documents,
        tags=topic_tags or None,
        is_foundational=True if state.hints.foundational_only else None,
        year_min=2024 if state.hints.recent_only else None,
    )
    tool_log = [
        ToolInvocation(
            name="corpus_metadata_lookup",
            args={
                "tags": topic_tags,
                "is_foundational": state.hints.foundational_only or None,
                "recent_only": state.hints.recent_only,
            },
            result_summary=f"matched {len(matches)} document(s)",
        )
    ]
    if not matches:
        return {
            "tool_log": tool_log,
            "answer": Answer(status=AnswerStatus.REFUSED, query=state.query, text=REFUSAL_TEXT),
        }

    # TOOL CALL: rank.
    ranked = rank_papers_for_reading_list(
        matches,
        foundational_first=state.hints.foundational_only,
        topic_tags=topic_tags,
    )[:7]
    tool_log.append(
        ToolInvocation(
            name="rank_papers_for_reading_list",
            args={
                "foundational_first": state.hints.foundational_only,
                "topic_tags": topic_tags,
            },
            result_summary=f"ranked {len(ranked)} paper(s)",
        )
    )

    citations = _matches_to_citations(ranked)
    text = _render_reading_list(citations, topic_tags)
    answer = Answer(
        status=AnswerStatus.ANSWERED,
        query=state.query,
        text=text,
        citations=citations,
        confidence=1.0,  # metadata-grounded; not retrieval-based confidence
        num_retrieved=len(ranked),
    )
    return {"tool_log": tool_log, "answer": answer}


# ---------- fallback (out_of_scope) ----------------------------------------


def fallback_node(state: AgentState, deps: NodeDeps) -> dict[str, Any]:
    """Polite decline for out-of-scope queries."""
    log.info("node_fallback", query=state.query)
    return {
        "answer": Answer(status=AnswerStatus.REFUSED, query=state.query, text=OUT_OF_SCOPE_TEXT)
    }


# ---------- helpers --------------------------------------------------------


def _matches_to_citations(matches: list[CorpusMetadataMatch]) -> list[Citation]:
    """Convert metadata matches into Citation objects for the FindPapers output."""
    citations: list[Citation] = []
    for i, m in enumerate(matches, start=1):
        if m.content_type == "arxiv_paper":
            arxiv_id = m.doc_id.replace("arxiv-", "", 1)
            source = f"arXiv:{arxiv_id}"
        elif m.content_type == "survey_blog":
            source = "Lil'Log"
        elif m.content_type == "course_chapter":
            source = "Hugging Face Learn"
        elif m.content_type == "lab_blog_post":
            source = m.doc_id.split("-")[0].capitalize()
        else:
            source = m.content_type
        section = "Foundational" if m.is_foundational else "Recent"
        citations.append(
            Citation(
                number=i,
                doc_id=m.doc_id,
                title=m.title,
                authors_display=format_authors(m.authors),
                year=m.year,
                source=source,
                section=section,
                url=m.source_url,
            )
        )
    return citations


def _render_reading_list(citations: list[Citation], topic_tags: list[str]) -> str:
    """Render the find-papers answer body (templated, not free-form)."""
    topic = ", ".join(topic_tags) if topic_tags else "the topic"
    lines = [
        f"Recommended reading on {topic} "
        "(ranked by foundational status, topical relevance, and recency):",
        "",
    ]
    for c in citations:
        lines.append(f"[{c.number}] **{c.title}** — {c.authors_display} ({c.year}, {c.source})")
    return "\n".join(lines)


def _doc_to_match(doc: DocumentMeta) -> CorpusMetadataMatch:
    return CorpusMetadataMatch(
        doc_id=doc.doc_id,
        title=doc.title,
        content_type=doc.content_type,
        year=doc.year,
        month=doc.month,
        authors=list(doc.authors),
        tags=list(doc.tags),
        is_foundational=doc.is_foundational,
        source_url=doc.source_url,
    )
