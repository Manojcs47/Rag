"""Prompt construction for the OpenAI-compatible generation backend (M2).

The system prompt enforces the grounding contract: answer only from the numbered
sources, attach a ``[n]`` marker to every factual sentence, and refuse if the
sources do not contain the answer. The refusal sentence is fixed so the pipeline
can recognise an LLM-side refusal and the eval harness (M4) can score it.
"""

from __future__ import annotations

from research_navigator.generate.citations import Citation
from research_navigator.retrieve.hybrid import RetrievedChunk

REFUSAL_TEXT = "I don't have enough relevant material in the corpus to answer this confidently."

SYSTEM_PROMPT = (
    "You are the AI Research Navigator, a citation-grounded assistant for AI/ML "
    "learners. You answer strictly from the numbered SOURCES provided in the user "
    "message and never use outside knowledge.\n\n"
    "Rules:\n"
    "1. Every factual sentence MUST end with one or more citation markers like [1] "
    "or [2][3], referring to the source(s) that support it.\n"
    "2. Use only the source numbers given. Never invent a source number.\n"
    "3. Do not present a source's wording as a direct quote; paraphrase in your own "
    "words and cite it.\n"
    "4. If the sources do not contain enough information to answer, reply with "
    f"exactly: {REFUSAL_TEXT}\n"
    "5. Be concise and accurate. Do not pad. Do not add a sources/references list — "
    "it is appended automatically."
)


def _format_source(number: int, chunk: RetrievedChunk, citation: Citation) -> str:
    header = f"[{number}] {citation.title} ({citation.source}, {citation.year})"
    if citation.section:
        header += f" — section: {citation.section}"
    body = chunk.text.strip()
    return f"{header}\n{body}"


def build_user_prompt(
    query: str,
    sources: list[tuple[Citation, RetrievedChunk]],
) -> str:
    """Assemble the user-turn prompt: the question plus the numbered source blocks."""
    blocks = [_format_source(c.number, chunk, c) for c, chunk in sources]
    joined = "\n\n".join(blocks)
    return (
        f"SOURCES:\n{joined}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer the question using only the sources above, citing each factual "
        "sentence with its source number(s)."
    )
