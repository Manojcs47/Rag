"""Generation: grounded answers with inline citations, or a graceful refusal.

Given the chunks retrieved for a query, this package (a) deduplicates them into a
Perplexity-style numbered citation block (one entry per source document, pointing
at its most relevant section), (b) generates an answer in which every factual
sentence carries an inline ``[n]`` marker, and (c) refuses when retrieval
confidence is too low — never fabricating an answer or a citation.
"""

from __future__ import annotations

from research_navigator.generate.citations import Citation, build_citations
from research_navigator.generate.generator import (
    ExtractiveGenerator,
    GeneratedAnswer,
    Generator,
    OpenAICompatibleGenerator,
    build_generator,
)
from research_navigator.generate.pipeline import (
    Answer,
    AnswerStatus,
    QueryEngine,
    answer_query,
)

__all__ = [
    "Answer",
    "AnswerStatus",
    "Citation",
    "ExtractiveGenerator",
    "GeneratedAnswer",
    "Generator",
    "OpenAICompatibleGenerator",
    "QueryEngine",
    "answer_query",
    "build_citations",
    "build_generator",
]
