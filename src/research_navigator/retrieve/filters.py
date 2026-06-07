"""Build Qdrant filter primitives from inferred query metadata (M2).

Filters are applied *inside* Qdrant (passed to each retrieval branch), never as a
post-hoc Python step — this is a hard requirement of the brief and the only way the
payload indexes created in M1 (B5) earn their keep.

Semantics:
  * ``year_min`` / ``year_max`` -> a single ``year`` range condition.
  * ``tags`` -> ``MatchAny`` (a document matches if it carries *any* inferred tag).
    OR-semantics protect recall when several tags are inferred from one question.
  * ``content_types`` -> ``MatchAny`` over ``content_type``.
  * ``is_foundational`` -> exact bool match.

Tags are separable (``include_tags=False``) so the query engine can relax to a
non-tag filter and retry rather than refuse when a mis-inferred tag empties the
candidate set (a documented, logged fallback — never a silent one).
"""

from __future__ import annotations

from qdrant_client import models

from research_navigator.retrieve.query import QueryAnalysis


def build_filter(analysis: QueryAnalysis, *, include_tags: bool = True) -> models.Filter | None:
    """Translate ``analysis`` into a Qdrant ``Filter`` (or ``None`` if unconstrained)."""
    must: list[models.FieldCondition] = []

    if analysis.year_min is not None or analysis.year_max is not None:
        must.append(
            models.FieldCondition(
                key="year",
                range=models.Range(gte=analysis.year_min, lte=analysis.year_max),
            )
        )

    if analysis.content_types:
        must.append(
            models.FieldCondition(
                key="content_type",
                match=models.MatchAny(any=list(analysis.content_types)),
            )
        )

    if analysis.is_foundational is not None:
        must.append(
            models.FieldCondition(
                key="is_foundational",
                match=models.MatchValue(value=analysis.is_foundational),
            )
        )

    if include_tags and analysis.tags:
        must.append(
            models.FieldCondition(key="tags", match=models.MatchAny(any=list(analysis.tags)))
        )

    return models.Filter(must=must) if must else None
