"""Per-content-type parsing: corpus files -> a clean :class:`ParsedDocument`.

Two parsers:
  * Markdown (HF course chapters, Lil'Log surveys, lab blogs) — strips the noise
    catalogued in the roadmap (synthetic headers, MDX components, ``[[anchors]]``,
    ``<!-- comments -->``, Lil'Log TOC / reading-time / nav / social cruft) while
    preserving section titles and fenced code blocks.
  * PDF (arXiv papers) via PyMuPDF — extracts text, separates the References
    section (excluded from retrieval, retained for citation lookup), and isolates
    the abstract as its own section.

The parser never raises on noisy input; it logs and degrades. A document that
yields zero sections is reported by the pipeline, not silently dropped.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from research_navigator.logging import get_logger

log = get_logger(__name__)

# --- shared regexes ---------------------------------------------------------
_CODE_FENCE = re.compile(r"(```.*?```|~~~.*?~~~)", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_HEADING_ANCHOR = re.compile(r"\[\[[^\]]*\]\]")  # ## Title[[slug]]
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Paired/known MDX or HTML block components whose *content* is non-prose.
_BLOCK_COMPONENTS = re.compile(
    r"<(Question|CourseFloatingBanner|Youtube|FrameworkSwitchCourse|Tip|"
    r"div|figure|iframe|table|hfoptions|hfoption)\b.*?(/>|</\1>)",
    re.DOTALL | re.IGNORECASE,
)
_SELF_CLOSING = re.compile(r"<[A-Za-z][A-Za-z0-9]*\b[^>]*/>")
_STRAY_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*\b[^>]*>")
_REFERENCES_HEADING = re.compile(r"(?im)^\s*#*\s*(references|citation|bibliography)\s*#*\s*$")
_BLANK_RUN = re.compile(r"\n{3,}")


class Section(BaseModel):
    """A retrievable section of a document with its heading hierarchy."""

    title: str
    level: int
    path: list[str] = Field(default_factory=list)  # ancestor titles + own title
    text: str


class ParsedDocument(BaseModel):
    """A document after cleaning, split into sections + optional abstract + refs."""

    doc_id: str
    content_type: str
    abstract: str | None = None
    sections: list[Section] = Field(default_factory=list)
    references: str = ""  # retained for citation lookup; excluded from retrieval


# --------------------------------------------------------------------------- #
# Markdown                                                                     #
# --------------------------------------------------------------------------- #
def _split_code(text: str) -> list[tuple[str, bool]]:
    """Split into (segment, is_code) parts so cleaning never touches code."""
    parts: list[tuple[str, bool]] = []
    last = 0
    for m in _CODE_FENCE.finditer(text):
        if m.start() > last:
            parts.append((text[last : m.start()], False))
        parts.append((m.group(0), True))
        last = m.end()
    if last < len(text):
        parts.append((text[last:], False))
    return parts


def _clean_prose(prose: str) -> str:
    """Strip MDX/HTML noise and heading anchors from a non-code segment."""
    prose = _HTML_COMMENT.sub("", prose)
    prose = _BLOCK_COMPONENTS.sub("", prose)
    prose = _SELF_CLOSING.sub("", prose)
    prose = _STRAY_TAG.sub("", prose)
    prose = _HEADING_ANCHOR.sub("", prose)
    return prose


def _strip_markdown_noise(text: str) -> str:
    """Apply :func:`_clean_prose` to prose segments, leaving code fences intact."""
    cleaned = "".join(seg if is_code else _clean_prose(seg) for seg, is_code in _split_code(text))
    return _BLANK_RUN.sub("\n\n", cleaned).strip()


def _drop_synthetic_header(text: str) -> str:
    """Remove the ``# Title / Source: / Author: ... \n---`` block prepended at fetch time."""
    lines = text.splitlines()
    for i, line in enumerate(lines[:12]):
        if line.strip() == "---":
            return "\n".join(lines[i + 1 :])
    return text


def _strip_lillog_cruft(text: str) -> str:
    """Remove Lil'Log-specific duplicated title, reading-time line, TOC, and trailing nav."""
    lines = text.splitlines()
    out: list[str] = []
    skipping_toc = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"(?i)^date:.*estimated reading time", stripped):
            continue
        if stripped.lower() == "table of contents":
            skipping_toc = True
            continue
        if skipping_toc:
            if stripped == "" or stripped.startswith("*"):
                continue
            skipping_toc = False
        out.append(line)
    body = "\n".join(out)
    # Trailing nav: tag-link list + prev/next + social share start at the tag block.
    body = re.split(r"\n\s*\*\s*\[[A-Za-z][^\]]*\]\(<https://[^>]*/tags/", body)[0]
    return body


def parse_markdown(doc_id: str, content_type: str, raw: str) -> ParsedDocument:
    """Parse a Markdown corpus file into a :class:`ParsedDocument`."""
    text = _drop_synthetic_header(raw)
    if content_type == "survey_blog":
        text = _strip_lillog_cruft(text)
    text = _strip_markdown_noise(text)

    body, references = _split_off_references(text)
    sections = _split_markdown_sections(body)
    if not sections:
        log.warning("no_sections_parsed", doc_id=doc_id, content_type=content_type)
    return ParsedDocument(
        doc_id=doc_id,
        content_type=content_type,
        sections=sections,
        references=references.strip(),
    )


def _split_off_references(text: str) -> tuple[str, str]:
    """Split text at the last References/Citation/Bibliography heading."""
    matches = list(_REFERENCES_HEADING.finditer(text))
    if not matches:
        return text, ""
    cut = matches[-1].start()
    return text[:cut], text[cut:]


def _split_markdown_sections(body: str) -> list[Section]:
    """Walk headings, tracking hierarchy; gather text under each heading.

    ``stack`` holds the live ancestor chain *including* the current heading, so a
    section's ``path`` is exactly the titles on the stack when its text is flushed.
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, title) chain including current heading
    buf: list[str] = []

    def flush() -> None:
        content = "\n".join(buf).strip()
        if not content:
            return
        if stack:
            level, title = stack[-1]
            path = [t for _, t in stack]
        else:  # content before the first heading
            level, title, path = 0, "Preamble", ["Preamble"]
        sections.append(Section(title=title, level=level, path=path, text=content))

    for line in body.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            title = m.group(2).strip() or "(untitled)"
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


# --------------------------------------------------------------------------- #
# PDF (arXiv)                                                                  #
# --------------------------------------------------------------------------- #
_PDF_SECTION = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})*)\s+([A-Z][A-Za-z].{0,80})$")
_PDF_SECTION_NUM = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})*)\s*$")
_PDF_SECTION_TITLE = re.compile(r"^[A-Z][A-Za-z][A-Za-z \-]{1,58}$")


def _extract_pdf_text(path: str) -> str:
    """Extract concatenated page text via PyMuPDF (imported lazily)."""
    import fitz

    doc = fitz.open(path)
    try:
        return "\n".join(doc[i].get_text() for i in range(doc.page_count))
    finally:
        doc.close()


def _extract_abstract(text: str) -> tuple[str | None, str]:
    """Pull the abstract (between 'Abstract' and the first section) out of ``text``."""
    m = re.search(
        r"(?is)\babstract\b\s*[:\-]?\s*(.+?)(?:\n\s*1\s+introduction\b|\bintroduction\b)", text
    )
    if not m:
        return None, text
    abstract = re.sub(r"\s+", " ", m.group(1)).strip()
    if len(abstract) < 40:  # likely a false hit
        return None, text
    return abstract, text[m.end(1) :]


def parse_pdf(doc_id: str, content_type: str, path: str) -> ParsedDocument:
    """Parse an arXiv PDF into a :class:`ParsedDocument`."""
    raw = _extract_pdf_text(path)
    body, references = _split_off_references(raw)
    abstract, body = _extract_abstract(body)

    sections: list[Section] = []
    cur_title = "Body"
    buf: list[str] = []

    def flush() -> None:
        content = re.sub(r"\n{3,}", "\n\n", "\n".join(buf)).strip()
        if content:
            sections.append(Section(title=cur_title, level=1, path=[cur_title], text=content))

    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        m = _PDF_SECTION.match(stripped)
        if m:
            flush()
            buf = []
            cur_title = f"{m.group(1)} {m.group(2).strip()}"
            i += 1
            continue
        # number on its own line, title on the next (common in arXiv extraction)
        num = _PDF_SECTION_NUM.match(stripped)
        if num and i + 1 < len(lines) and _PDF_SECTION_TITLE.match(lines[i + 1].strip()):
            flush()
            buf = []
            cur_title = f"{num.group(1)} {lines[i + 1].strip()}"
            i += 2
            continue
        buf.append(line)
        i += 1
    flush()

    if not sections and not abstract:
        log.warning("no_sections_parsed", doc_id=doc_id, content_type=content_type)
    return ParsedDocument(
        doc_id=doc_id,
        content_type=content_type,
        abstract=abstract,
        sections=sections,
        references=references.strip(),
    )


def parse_document(doc_id: str, content_type: str, path: str) -> ParsedDocument:
    """Dispatch to the right parser based on file extension."""
    if path.lower().endswith(".pdf"):
        return parse_pdf(doc_id, content_type, path)
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    return parse_markdown(doc_id, content_type, raw)
