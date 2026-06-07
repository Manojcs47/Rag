"""Unit tests for citation construction, deduplication, and rendering (M2)."""

from __future__ import annotations

from research_navigator.generate.citations import (
    build_citations,
    format_authors,
    source_label,
)
from research_navigator.retrieve.hybrid import RetrievedChunk


def _chunk(
    doc_id: str,
    *,
    content_type: str = "arxiv_paper",
    title: str = "A Paper",
    authors: list[str] | None = None,
    year: int = 2024,
    source_url: str = "https://arxiv.org/abs/2305.14314",
    section_path: list[str] | None = None,
    kind: str = "body",
    score: float = 1.0,
) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id,
        content_type=content_type,
        title=title,
        authors=authors or ["Ada Lovelace", "Bob Recursor", "Cara Turing"],
        year=year,
        source_url=source_url,
        section_title=(section_path or ["Body"])[-1],
        section_path=section_path or ["Body"],
        kind=kind,
        text="Some grounded sentence about the method.",
        score=score,
    )


def test_format_authors_rules() -> None:
    assert format_authors(["Solo"]) == "Solo"
    assert format_authors(["A", "B"]) == "A and B"
    assert format_authors(["A", "B", "C"]) == "A et al."
    assert format_authors([]) == ""


def test_source_label_per_content_type() -> None:
    assert source_label(_chunk("arxiv-2305.14314")) == "arXiv:2305.14314"
    assert source_label(_chunk("lillog-x", content_type="survey_blog")) == "Lil'Log"
    assert (
        source_label(_chunk("hf-nlp-ch01", content_type="course_chapter")) == "Hugging Face Learn"
    )
    assert (
        source_label(_chunk("anthropic-mapping-mind-2024-05", content_type="lab_blog_post"))
        == "Anthropic"
    )


def test_dedup_collapses_same_document_to_one_citation() -> None:
    chunks = [
        _chunk("d1", section_path=["Intro"], score=0.9),
        _chunk("d1", section_path=["Method"], score=0.5),  # same doc, lower score
        _chunk("d2", title="Other", source_url="https://arxiv.org/abs/2401.00001", score=0.7),
    ]
    citations, doc_to_number = build_citations(chunks)
    assert len(citations) == 2  # d1 collapsed
    # Numbering by best-chunk score: d1 (0.9) -> 1, d2 (0.7) -> 2.
    assert doc_to_number == {"d1": 1, "d2": 2}
    # The d1 citation points at its most relevant section (Intro, score 0.9).
    assert citations[0].section == "Intro"


def test_abstract_section_label() -> None:
    citations, _ = build_citations([_chunk("d1", kind="abstract")])
    assert citations[0].section == "Abstract"


def test_citation_render_contains_marker_and_url() -> None:
    citations, _ = build_citations([_chunk("arxiv-2305.14314")])
    rendered = citations[0].render()
    assert rendered.startswith("[1] ")
    assert "arXiv:2305.14314" in rendered
    assert "https://arxiv.org/abs/2305.14314" in rendered
    assert "et al." in rendered
