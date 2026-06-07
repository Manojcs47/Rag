# Ingestion pipeline (M1)

The ingestion pipeline turns the sealed corpus into retrievable, citable chunks in
Qdrant. It is a linear pipeline with a strict idempotency contract, orchestrated by
`research_navigator.ingest.pipeline`.

```
manifest.json
     |
     v
[manifest]  load + validate (pydantic)            ingest/manifest.py
     |
     v
[parse]     PDF (PyMuPDF) | Markdown (regex)       ingest/parse.py
     |        - strip MDX/anchors/comments/nav cruft
     |        - isolate abstract (papers)
     |        - split off References (kept, not retrieved)
     |        - build a section tree with heading hierarchy
     v
[chunk]     section-bounded, content-type budgets   ingest/chunk.py
     |        - abstract = own chunk; code atomic
     |        - token-bounded packing + overlap
     |        - content_hash + deterministic point_id
     v
[embed]     dense (sentence-transformers | offline)  ingest/embed.py
     |        sparse (TF vectors; IDF applied by Qdrant)
     v
[store]     idempotent upsert into Qdrant            ingest/qdrant_store.py
              - named dense (cosine) + sparse (IDF) vectors
              - payload = all manifest fields + per-chunk fields
              - payload indexes for M2 filtering
```

## Modules

- **`manifest.py`** — loads `manifest.json` into strict pydantic models. An
  unexpected schema is a loud failure, not a silent `KeyError` downstream.
- **`parse.py`** — two parsers behind one `parse_document` dispatcher. Markdown
  parsing protects fenced code while stripping noise (synthetic headers, MDX
  components, `[[anchors]]`, HTML comments, Lil'Log TOC/reading-time/nav). PDF
  parsing (PyMuPDF) extracts text, isolates the abstract, splits off References, and
  detects numbered section headers (including the common "number on one line, title
  on the next" arXiv extraction artifact). Parsers degrade rather than raise.
- **`chunk.py`** — `chunk_document` enforces ADR-0003: abstract-as-chunk,
  section-bounded packing to a per-type token budget with overlap, atomic code
  fences, references excluded. Each chunk carries a `content_hash` and exposes a
  deterministic `point_id` (ADR-0005).
- **`embed.py`** — `build_embedders(settings)` returns a dense + sparse embedder pair
  behind `Protocol`s. Production dense = `sentence-transformers`; CI/tests dense =
  deterministic hashing embedder (no download). Sparse = log-TF over a hashed vocab;
  Qdrant supplies IDF (ADR-0004).
- **`qdrant_store.py`** — collection schema (named dense + sparse vectors, payload
  indexes), idempotency helpers (`existing_point_ids`), `upsert_chunks`,
  `delete_points`, and `stats`.
- **`pipeline.py`** — orchestration + the idempotency diff. `ingest_corpus`,
  `validate_corpus`, `reindex_corpus`. Per-document errors are reported, never
  swallowed; the run continues and surfaces them in the `IngestReport`.
- **`cli/ingest.py`** — a thin argparse front end (`ingest` / `validate` / `reindex`
  / `stats`). No business logic; JSON to stdout, logs to stderr, non-zero exit on any
  document error.

## Idempotency contract (B4)

A chunk's Qdrant id is `uuid5(NAMESPACE, "{doc_id}:{chunk_index}:{content_hash}")`.
On each run, per document, the pipeline computes the desired id set, reads the
existing id set from Qdrant, then upserts only new ids, deletes only stale ids, and
skips the document entirely when the sets match. Therefore:

- Re-ingesting an unchanged corpus writes nothing.
- Editing one document re-embeds/upserts only its changed chunks and deletes only its
  orphaned chunks; every other document is untouched.

Both properties are covered by integration tests in
`tests/test_pipeline_integration.py`.

## Configuration

Everything tunable lives in `research_navigator.config.Settings` (env prefix `RN_`,
`.env` honoured, nested chunk params via the `__` delimiter). Nothing in `src/`
hardcodes a path, model, or threshold. See `.env.example`.

## Testing without a server or network

Tests point `RN_QDRANT_URL` at `:memory:` (in-process Qdrant) and set
`RN_USE_OFFLINE_EMBEDDER=1` (deterministic hashing embedder). A tiny fixture corpus
carries the same noise patterns as the real one, so parsing/chunking are exercised
against realistic input. The full pipeline — including the idempotency diff — runs
end to end with zero external dependencies.

> Note: Qdrant's local (`:memory:`) mode accepts the payload-index calls but treats
> them as no-ops (it logs a warning). Indexes take effect against a real server,
> which is what M2 retrieval will use.
