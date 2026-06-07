"""Citation construction + rendering (M2).

Turns retrieved chunks into a deduplicated, numbered citation block in the style the
brief specifies: title, authors (first author *et al.* for >= 3), year, source label
(e.g. ``arXiv:2305.14314``, ``Lil'Log``, ``Hugging Face Learn``), section, and a
direct URL.

Deduplication: multiple chunks from one document collapse into a single citation
whose ``section`` points at the highest-scoring (most relevant) chunk from that
document. Citation numbers are assigned by best-chunk score, so ``[1]`` is the most
relevant source.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from research_navigator.retrieve.hybrid import RetrievedChunk

_ARXIV_ABS = re.compile(r"/abs/([^/?#]+)")
_ARXIV_DOCID = re.compile(r"^arxiv-(.+)$", re.IGNORECASE)

# Lab-blog source labels keyed by doc_id prefix (falls back to the URL host).
_LAB_LABELS: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "deepmind": "Google DeepMind",
    "google": "Google DeepMind",
}


class Citation(BaseModel):
    """One numbered citation entry mapping a marker ``[number]`` to a source."""

    number: int
    doc_id: str
    title: str
    authors_display: str
    year: int
    source: str
    section: str
    url: str

    def render(self) -> str:
        """One-line human-readable citation, e.g. used in CLI/Markdown output."""
        bits = [f"[{self.number}] {self.title}"]
        if self.authors_display:
            bits.append(self.authors_display)
        bits.append(str(self.year))
        bits.append(self.source)
        if self.section:
            bits.append(f"§ {self.section}")
        line = " — ".join(bits)
        return f"{line} — {self.url}" if self.url else line


def format_authors(authors: list[str]) -> str:
    """Brief's rule: 1 -> name; 2 -> 'A and B'; >= 3 -> 'A et al.'."""
    cleaned = [a.strip() for a in authors if a and a.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{cleaned[0]} et al."


def source_label(chunk: RetrievedChunk) -> str:
    """Human-friendly source label per the brief's examples."""
    ct = chunk.content_type
    if ct == "arxiv_paper":
        arxiv_id = ""
        if (m := _ARXIV_ABS.search(chunk.source_url)) or (m := _ARXIV_DOCID.match(chunk.doc_id)):
            arxiv_id = m.group(1)
        return f"arXiv:{arxiv_id}" if arxiv_id else "arXiv"
    if ct == "survey_blog":
        return "Lil'Log"
    if ct == "course_chapter":
        return "Hugging Face Learn"
    if ct == "lab_blog_post":
        prefix = chunk.doc_id.split("-", 1)[0].lower()
        if prefix in _LAB_LABELS:
            return _LAB_LABELS[prefix]
        host = re.sub(r"^https?://(www\.)?", "", chunk.source_url).split("/", 1)[0]
        return host or "Lab blog"
    return ct or "Source"


def _section_for(chunk: RetrievedChunk) -> str:
    """Render the chunk's section, preferring the full heading path."""
    if chunk.kind == "abstract":
        return "Abstract"
    if chunk.section_path:
        return "  > ".join(chunk.section_path)
    return chunk.section_title


def build_citations(chunks: list[RetrievedChunk]) -> tuple[list[Citation], dict[str, int]]:
    """Deduplicate chunks by document into numbered citations.

    Returns:
        ``(citations, doc_to_number)`` — the ordered citation list and a map from
        ``doc_id`` to its assigned citation number, so a generator can attach the
        right marker to a sentence drawn from a given document.
    """
    # Best (highest-scoring) chunk per document, preserving first-seen order as the
    # tie-break (Qdrant already returns chunks best-first).
    best: dict[str, RetrievedChunk] = {}
    order: list[str] = []
    for chunk in chunks:
        if chunk.doc_id not in best:
            best[chunk.doc_id] = chunk
            order.append(chunk.doc_id)
        elif chunk.score > best[chunk.doc_id].score:
            best[chunk.doc_id] = chunk

    # Number by descending best-chunk score (stable on ties via original order).
    ranked = sorted(order, key=lambda d: (-best[d].score, order.index(d)))

    citations: list[Citation] = []
    doc_to_number: dict[str, int] = {}
    for i, doc_id in enumerate(ranked, start=1):
        chunk = best[doc_id]
        doc_to_number[doc_id] = i
        citations.append(
            Citation(
                number=i,
                doc_id=doc_id,
                title=chunk.title,
                authors_display=format_authors(chunk.authors),
                year=chunk.year,
                source=source_label(chunk),
                section=_section_for(chunk),
                url=chunk.source_url,
            )
        )
    return citations, doc_to_number


def render_citation_block(citations: list[Citation]) -> str:
    """Render citations as a Markdown 'Sources' list (one line per entry)."""
    if not citations:
        return ""
    lines = ["**Sources**"]
    lines.extend(c.render() for c in citations)
    return "\n".join(lines)
