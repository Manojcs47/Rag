"""Unit tests for parsing: noise stripping, section trees, references, code survival."""

from __future__ import annotations

from research_navigator.ingest.parse import (
    ParsedDocument,
    parse_document,
    parse_markdown,
)


def test_parse_markdown_strips_mdx_and_anchors(hf_md: str) -> None:
    parsed = parse_markdown("hf-nlp-ch09", "course_chapter", hf_md)
    blob = "\n".join(s.text for s in parsed.sections)
    # MDX components, heading anchors and HTML comments must be gone.
    assert "CourseFloatingBanner" not in blob
    assert "<Youtube" not in blob
    assert "[[introduction]]" not in blob
    assert "<!-- Section" not in blob


def test_parse_markdown_drops_synthetic_header(hf_md: str) -> None:
    parsed = parse_markdown("hf-nlp-ch09", "course_chapter", hf_md)
    blob = "\n".join(s.text for s in parsed.sections)
    # The "Source: https://..." synthetic preamble line is removed.
    assert "Source: https://huggingface.co" not in blob


def test_parse_markdown_preserves_code_block(hf_md: str) -> None:
    parsed = parse_markdown("hf-nlp-ch09", "course_chapter", hf_md)
    blob = "\n".join(s.text for s in parsed.sections)
    # Code content must survive cleaning intact.
    assert "from transformers import pipeline" in blob
    assert 'pipeline("sentiment-analysis")' in blob


def test_parse_markdown_builds_section_titles(hf_md: str) -> None:
    parsed = parse_markdown("hf-nlp-ch09", "course_chapter", hf_md)
    titles = {s.title for s in parsed.sections}
    assert "Introduction" in titles
    assert "Setup" in titles


def test_section_path_tracks_hierarchy(hf_md: str) -> None:
    parsed = parse_markdown("hf-nlp-ch09", "course_chapter", hf_md)
    setup = next(s for s in parsed.sections if s.title == "Setup")
    # Setup is an H2 nested under the H1 "Introduction".
    assert setup.path[-1] == "Setup"
    assert "Introduction" in setup.path


def test_lillog_cruft_stripped(lillog_md: str) -> None:
    parsed = parse_markdown("lillog-prompt-eng-2023-03", "survey_blog", lillog_md)
    blob = "\n".join(s.text for s in parsed.sections)
    assert "Estimated Reading Time" not in blob
    assert "Table of Contents" not in blob
    # Trailing tag-nav links are stripped.
    assert "lilianweng.github.io/tags/" not in blob


def test_references_split_off(lillog_md: str) -> None:
    parsed = parse_markdown("lillog-prompt-eng-2023-03", "survey_blog", lillog_md)
    blob = "\n".join(s.text for s in parsed.sections)
    # References are excluded from retrievable sections...
    assert "A Paper." not in blob
    # ...but retained for citation lookup.
    assert parsed.references != ""
    assert "References" in parsed.references


def test_parse_document_dispatches_markdown(corpus_dir, hf_md: str) -> None:
    path = corpus_dir / "documents" / "hf-learn" / "hf-nlp-ch09.md"
    parsed = parse_document("hf-nlp-ch09", "course_chapter", str(path))
    assert isinstance(parsed, ParsedDocument)
    assert parsed.sections


def test_empty_document_yields_no_sections() -> None:
    parsed = parse_markdown("empty", "survey_blog", "# Title\n\n---\n")
    assert parsed.sections == []


# --- PDF path (generate a tiny real PDF with PyMuPDF) -----------------------
def _make_pdf(path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    text = (
        "Attention Is All You Need\n\n"
        "Abstract\n"
        "We propose the Transformer, a new architecture based solely on "
        "attention mechanisms, dispensing with recurrence entirely.\n\n"
        "1 Introduction\n"
        "Recurrent networks process tokens sequentially which limits parallelism.\n\n"
        "2 Background\n"
        "Self-attention relates different positions of a single sequence.\n\n"
        "References\n"
        "[1] Some Author. A cited paper. 2016.\n"
    )
    page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()


def test_parse_pdf_extracts_abstract_and_sections(tmp_path) -> None:
    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    parsed = parse_document("arxiv-test", "arxiv_paper", str(pdf))
    assert parsed.abstract is not None
    assert "Transformer" in parsed.abstract
    # Section headers detected from the numbered lines.
    titles = " ".join(s.title for s in parsed.sections)
    assert "Introduction" in titles
    # References separated out of retrievable text.
    body = "\n".join(s.text for s in parsed.sections)
    assert "cited paper" not in body
    assert parsed.references != ""
