"""Tool helpers invoked by agent nodes (M3).

These are structured, typed functions. Nodes call them explicitly and record
the call in :attr:`AgentState.tool_log`, satisfying the brief's "at least one
tool call within an agent node" requirement while keeping every step
inspectable and unit-testable.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from research_navigator.ingest.manifest import DocumentMeta


class CorpusMetadataMatch(BaseModel):
    """One document surfaced by a metadata lookup."""

    doc_id: str
    title: str
    content_type: str
    year: int
    month: int | None
    authors: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    is_foundational: bool = False
    source_url: str = ""


class DateRange(BaseModel):
    """A year-bounded date range."""

    year_min: int
    year_max: int | None = None


def corpus_metadata_lookup(
    documents: list[DocumentMeta],
    *,
    tags: list[str] | None = None,
    content_types: list[str] | None = None,
    is_foundational: bool | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    doc_id: str | None = None,
) -> list[CorpusMetadataMatch]:
    """Filter the manifest by metadata fields.

    All constraints AND together. ``tags`` use OR semantics (a doc matches if it
    carries any requested tag), mirroring the M2 retrieval filter (ADR style).
    """
    tag_set = set(tags or [])
    type_set = set(content_types or [])
    out: list[CorpusMetadataMatch] = []
    for doc in documents:
        if doc_id is not None and doc.doc_id != doc_id:
            continue
        if tag_set and not (tag_set & set(doc.tags)):
            continue
        if type_set and doc.content_type not in type_set:
            continue
        if is_foundational is not None and doc.is_foundational != is_foundational:
            continue
        if year_min is not None and doc.year < year_min:
            continue
        if year_max is not None and doc.year > year_max:
            continue
        out.append(_to_match(doc))
    return out


def date_math(months_back: int = 12, today: date | None = None) -> DateRange:
    """Return a year-floor cutoff ``months_back`` months before ``today``.

    Used by RecentDevelopments. Year-only because manifest ``month`` is nullable
    for course chapters (Roadmap §0 finding #5), so floor to the calendar year.
    """
    today = today or date.today()
    total_months = today.year * 12 + (today.month - 1) - months_back
    floor_year = total_months // 12
    return DateRange(year_min=floor_year)


def rank_papers_for_reading_list(
    matches: list[CorpusMetadataMatch],
    *,
    foundational_first: bool = False,
    topic_tags: list[str] | None = None,
) -> list[CorpusMetadataMatch]:
    """Rank candidates for a 'find papers' recommendation.

    Decision per ADR-0008: ``citation_count`` is null for every document in the
    sealed corpus (Roadmap §0 finding #4), so we rank on
    (``is_foundational``, tag-match-count, ``year``) — not on citation count.
    Ties broken by year descending.
    """
    topic_set = set(topic_tags or [])

    def score(m: CorpusMetadataMatch) -> tuple[int, int, int]:
        foundational_bit = (1 if m.is_foundational else 0) if foundational_first else 0
        tag_match = len(set(m.tags) & topic_set)
        return (foundational_bit, tag_match, m.year)

    return sorted(matches, key=score, reverse=True)


def _to_match(doc: DocumentMeta) -> CorpusMetadataMatch:
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
