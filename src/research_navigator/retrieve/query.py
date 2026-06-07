"""Query understanding: free text -> inferable metadata filters (M2).

The goal here is *filter inference*, not intent classification — the LangGraph
``Router`` (M3) decides the route. M2 only needs the structured signals that let
retrieval narrow the candidate set inside Qdrant:

  * **Year bounds** from words like "recent"/"latest" or explicit years
    ("in 2024", "since 2023", "before 2020").
  * **Tags** by matching corpus-vocabulary tags (and a few natural-language
    aliases) against the query.
  * **Content-type** hints ("survey", "course", "paper", "blog post").
  * **Foundational** hints ("seminal", "foundational", "classic").

Everything is deterministic and dependency-free so it is trivially unit-testable
and reproducible (M5). The known-tag vocabulary is *passed in* by the caller
(derived from the manifest), keeping the corpus a variable rather than a constant.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

# Documented corpus tag vocabulary (Data Corpus Details §"Tag vocabulary").
# Used as the default when a caller does not supply manifest-derived tags.
DEFAULT_TAG_VOCABULARY: frozenset[str] = frozenset(
    {
        "transformers",
        "attention",
        "pretraining",
        "fine_tuning",
        "instruction_tuning",
        "prompting",
        "chain_of_thought",
        "reasoning",
        "tool_use",
        "LLM",
        "MoE",
        "RAG",
        "retrieval",
        "agents",
        "alignment",
        "RLHF",
        "RLAIF",
        "DPO",
        "preference_optimization",
        "safety",
        "interpretability",
        "efficiency",
        "quantization",
        "long_context",
        "evaluation",
        "benchmark",
        "multimodal",
        "vision_language",
        "open_models",
        "scaling",
        "inference_optimization",
        "survey",
        "architecture",
        "few_shot",
        "RL",
    }
)

# Natural-language phrases that map onto a canonical tag. Keys are lower-cased and
# matched as whole words/phrases; values must exist in the tag vocabulary to apply.
_TAG_ALIASES: dict[str, str] = {
    "chain of thought": "chain_of_thought",
    "chain-of-thought": "chain_of_thought",
    "cot": "chain_of_thought",
    "preference optimization": "preference_optimization",
    "preference optimisation": "preference_optimization",
    "fine tuning": "fine_tuning",
    "fine-tuning": "fine_tuning",
    "instruction tuning": "instruction_tuning",
    "long context": "long_context",
    "tool use": "tool_use",
    "open weight": "open_models",
    "open-weight": "open_models",
    "open source models": "open_models",
    "mixture of experts": "MoE",
    "retrieval augmented generation": "RAG",
    "retrieval-augmented generation": "RAG",
    "large language model": "LLM",
    "large language models": "LLM",
    "agent": "agents",
    "reinforcement learning": "RL",
    "vision language": "vision_language",
    "few shot": "few_shot",
    "few-shot": "few_shot",
}

# Words signalling a recency intent without an explicit year.
_RECENCY_WORDS = re.compile(
    r"\b(recent|recently|latest|newest|new|lately|cutting[- ]edge|state[- ]of[- ]the[- ]art"
    r"|this year|last year|nowadays|these days)\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_SINCE = re.compile(r"\b(?:since|after|from)\s+((?:19|20)\d{2})\b", re.IGNORECASE)
_BEFORE = re.compile(r"\b(?:before|prior to|up to|until)\s+((?:19|20)\d{2})\b", re.IGNORECASE)
_IN_YEAR = re.compile(r"\b(?:in|during|published in)\s+((?:19|20)\d{2})\b", re.IGNORECASE)

# Content-type hints (phrase -> manifest content_type).
_CONTENT_TYPE_HINTS: dict[str, str] = {
    "survey": "survey_blog",
    "surveys": "survey_blog",
    "course": "course_chapter",
    "chapter": "course_chapter",
    "tutorial": "course_chapter",
    "lab blog": "lab_blog_post",
    "blog post": "lab_blog_post",
    "paper": "arxiv_paper",
    "papers": "arxiv_paper",
    "arxiv": "arxiv_paper",
}
_FOUNDATIONAL_WORDS = re.compile(
    r"\b(foundational|seminal|classic|landmark|canonical|original|pioneering)\b",
    re.IGNORECASE,
)


class QueryAnalysis(BaseModel):
    """Structured, inferable filters extracted from a learner's question.

    Every field is optional/empty by default; an empty analysis means "no metadata
    constraint", i.e. search the whole collection.
    """

    raw_query: str
    year_min: int | None = None
    year_max: int | None = None
    tags: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=list)
    is_foundational: bool | None = None
    wants_recent: bool = False

    @property
    def has_filters(self) -> bool:
        """True if any metadata constraint was inferred."""
        return bool(
            self.year_min is not None
            or self.year_max is not None
            or self.tags
            or self.content_types
            or self.is_foundational is not None
        )


def _infer_years(text: str, recent_year_floor: int) -> tuple[int | None, int | None, bool]:
    """Return ``(year_min, year_max, wants_recent)`` from the query text."""
    year_min: int | None = None
    year_max: int | None = None

    if m := _IN_YEAR.search(text):
        y = int(m.group(1))
        return y, y, False

    if m := _SINCE.search(text):
        year_min = int(m.group(1))
    if m := _BEFORE.search(text):
        year_max = int(m.group(1))

    wants_recent = bool(_RECENCY_WORDS.search(text))
    if wants_recent and year_min is None and year_max is None:
        year_min = recent_year_floor

    # A bare explicit year (no since/before/in qualifier) becomes a lower bound.
    if year_min is None and year_max is None and not wants_recent:
        years = [int(m.group(0)) for m in _YEAR.finditer(text)]
        if years:
            year_min = min(years)

    return year_min, year_max, wants_recent


def _infer_tags(text_lower: str, known_tags: frozenset[str]) -> list[str]:
    """Match vocabulary tags + aliases present in the (lower-cased) query."""
    found: list[str] = []
    seen: set[str] = set()

    def add(tag: str) -> None:
        if tag in known_tags and tag not in seen:
            seen.add(tag)
            found.append(tag)

    # Multi-word aliases first (longest phrases win before single-token matches).
    for phrase, tag in sorted(_TAG_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(phrase)}\b", text_lower):
            add(tag)

    # Direct vocabulary matches. Underscores in tags map to spaces or hyphens in prose.
    for tag in known_tags:
        variants = {tag.lower(), tag.lower().replace("_", " "), tag.lower().replace("_", "-")}
        if any(re.search(rf"\b{re.escape(v)}\b", text_lower) for v in variants):
            add(tag)

    return found


def _infer_content_types(text_lower: str) -> list[str]:
    out: list[str] = []
    for phrase, ctype in _CONTENT_TYPE_HINTS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", text_lower) and ctype not in out:
            out.append(ctype)
    return out


def analyze_query(
    query: str,
    *,
    known_tags: frozenset[str] | None = None,
    recent_year_floor: int = 2024,
) -> QueryAnalysis:
    """Extract inferable metadata filters from ``query``.

    Args:
        query: The learner's natural-language question.
        known_tags: Tag vocabulary to match against (defaults to the documented
            corpus vocabulary). Pass the manifest-derived tag set to stay correct
            when the corpus is swapped.
        recent_year_floor: Lower-bound year an unqualified "recent" maps to.
    """
    vocab = known_tags if known_tags is not None else DEFAULT_TAG_VOCABULARY
    text_lower = query.lower()

    year_min, year_max, wants_recent = _infer_years(query, recent_year_floor)
    is_foundational = True if _FOUNDATIONAL_WORDS.search(query) else None

    return QueryAnalysis(
        raw_query=query,
        year_min=year_min,
        year_max=year_max,
        tags=_infer_tags(text_lower, vocab),
        content_types=_infer_content_types(text_lower),
        is_foundational=is_foundational,
        wants_recent=wants_recent,
    )
