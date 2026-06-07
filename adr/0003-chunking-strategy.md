# ADR-0003 — Section-bounded, content-type-aware chunking

- **Status:** Accepted
- **Date:** 2025 (M1)
- **Deciders:** project owner (intern) + reviewer
- **Relates to:** B1 (parsing), B2 (chunking), ADR-0005 (idempotency)

## Context

Chunking determines what a retriever can return and what a citation can point at.
The corpus is heterogeneous: arXiv PDFs (abstract + numbered sections + references),
HuggingFace course chapters (multiple concatenated pages, MDX noise, meaningful code
blocks), Lil'Log surveys (header/TOC/nav cruft, references), and lab blog posts
(variable HTML→Markdown quality). A single fixed-size sliding window over raw text
would split mid-section, shatter code blocks, and pollute the index with navigation
and reference cruft.

We also need token counting that is **deterministic** so that re-running ingestion
produces identical chunks (ADR-0005 depends on this), without pulling a specific
tokenizer as a hard dependency.

## Decision

Chunk **per content type**, bounded by document structure:

1. **Abstract is its own chunk** (papers) — it is the highest-value retrieval unit.
2. **Never cross section boundaries.** Parsing yields a section tree with its
   heading hierarchy (`section_path`); chunking packs text *within* a section up to
   a token budget, with a small token-bounded overlap so adjacent chunks share
   context.
3. **Code fences are atomic.** A fenced block is never split; if it exceeds the
   budget it becomes its own chunk rather than being broken mid-syntax.
4. **References are excluded from retrieval** but retained on the parsed document for
   later citation lookup — they are noise for semantic search but needed for grounding.
5. **Per-type token budgets** (configurable, nested env vars):
   - `arxiv_paper`: 512 / 64 overlap
   - `course_chapter`: 384 / 48 (denser, more headings)
   - `survey_blog`, `lab_blog_post`: 512 / 64
6. **Deterministic token counter:** a dependency-free `\w+|[^\w\s]` regex
   approximation. Model-independent and stable across runs.
7. Oversized prose blocks split on sentence boundaries first, then a hard
   word-window fallback, so no prose chunk exceeds its budget. Tiny trailing chunks
   are merged back into their predecessor.

## Consequences

- Chunks align with human-meaningful units (a section, the abstract, a code block),
  which improves both retrieval relevance and citation precision in later milestones.
- The token counter is an approximation, not the embedding model's tokenizer, so a
  chunk's true token count under `bge` differs slightly — acceptable, since budgets
  are guidance and the model truncates safely.
- Pathological inputs (e.g. a Markdown table with no sentence boundaries) can
  produce a chunk marginally over budget via the word-window fallback; rare and
  documented, never catastrophic.
- Budgets and overlaps are tunable per type without code changes (M2 can revisit).

## Alternatives considered

- **Fixed-size sliding window** over raw text — simplest, but ignores structure,
  splits code, and indexes cruft. Rejected.
- **Recursive character splitter (LangChain-style)** — structure-aware-ish but still
  character-based and not section/code-aware in the way this corpus needs. Rejected
  in favour of an explicit, testable, content-type-aware splitter.
- **Model-tokenizer token counting** — most accurate, but couples chunking to a
  specific model and adds a dependency; rejected for determinism and decoupling.
