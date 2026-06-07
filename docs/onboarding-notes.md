# M0 Onboarding — Corpus Structure & Chunking Implications

Observations from skimming the five corpus buckets. These feed the chunking ADR (M1).

## 1. arXiv papers — foundational (10) & recent (20)  [content_type: arxiv_paper]
- Native PDF, often multi-column academic layout, with: abstract, numbered
  sections, figures/tables, footnotes, and a References section at the end.
- Foundational vs recent differ only by `is_foundational` + era; they parse identically.
- Implications: extract text with PyMuPDF; read columns in correct order; treat the
  **abstract as its own chunk**; split on section headings; **skip or summarise
  figures/tables**; **exclude References from retrieval but keep them for citation
  lookup**.

## 2. Hugging Face course chapters (12)  [content_type: course_chapter]
- One `.md` file concatenates several course *pages*, each with its own top-level
  `#` heading — so a single file has multiple H1 sections plus nested H2/H3.
- Contains MDX noise: `<CourseFloatingBanner .../>`, `<Youtube id=.../>`,
  heading anchors like `## Title[[slug]]`, and `<!-- Section 1.1 -->` comments.
  Code blocks are present and meaningful.
- Implications: strip MDX components, anchors, and HTML comments; **preserve the
  heading hierarchy** for `section_title`; **keep code blocks intact**.

## 3. Lil'Log surveys (5)  [content_type: survey_blog]
- Header block (Author/Published/Source), a duplicated title, a "Table of Contents",
  and a reading-time line at the top. Body is well-structured with H2/H3 and inline
  reference markers like `[12]`. Ends with a numbered References list, then trailing
  cruft: tag links, prev/next navigation, and social-share buttons.
- Implications: strip the top header/TOC/reading-time and the **trailing nav/share
  block**; keep references for citation lookup; section-aware chunking works well here.

## 4. Lab blog posts (3)  [content_type: lab_blog_post]
- Fetched + converted to Markdown by `complete_corpus.py`. Each begins with a
  generated header (`# title`, `Source:`, `Publisher:`, `---`) followed by the
  extracted body. Extraction quality varies (trafilatura is cleaner than html2text)
  and some residual nav text may remain.
- Implications: use the generated header as metadata, not body content; expect to
  clean variable HTML-to-Markdown artefacts.

## Cross-cutting chunking decisions (to finalise in the ADR)
- Abstract = standalone chunk (papers).
- Split on section boundaries, then size-bound within a section (token budget +
  small overlap).
- Keep code blocks whole; never split mid-block.
- Exclude References from retrieval; retain for citation lookup.
- Every chunk carries: section_title, section_index, chunk_index, content_hash,
  plus all document-level manifest fields.
- `month` is null for course chapters and `citation_count` is null for all docs —
  downstream sorting/ranking must tolerate nulls.
