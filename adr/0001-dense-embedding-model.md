# ADR-0001 — Local open-source dense embeddings via sentence-transformers

- **Status:** Accepted
- **Date:** 2025 (M1)
- **Deciders:** project owner (intern) + reviewer
- **Relates to:** B2 (embedding), ADR-0004 (sparse retrieval)

## Context

M1 needs a dense embedding model to turn chunks into vectors for semantic search.
The two realistic families are (a) a hosted API (OpenAI `text-embedding-3`, Cohere,
Voyage) or (b) a local open-source model run via `sentence-transformers`.

Constraints that shaped the call:

- The corpus is sealed / non-redistributable; sending its text to a third-party
  embedding API is undesirable on principle and creates a data-handling question we
  would rather not carry.
- The grading environment and CI must run without secrets, without paid quota, and
  ideally without network access at all.
- Reproducibility (M5): the same input must produce the same vectors on re-run.
- Cost: re-embedding ~3–4k chunks repeatedly during development should be free.

## Decision

Use **local open-source dense models through `sentence-transformers`**, default
`BAAI/bge-small-en-v1.5` (384-dim, strong MTEB scores for its size, CPU-friendly).
The model id, dimensionality, device, batch size, and passage prefix are all
configurable (`RN_DENSE_MODEL`, `RN_DENSE_DIM`, …) so swapping to `bge-base`, an
`e5` model, or a GPU is a config change, not a code change.

We also ship a deterministic, dependency-free **offline hashing embedder**
(`HashingDenseEmbedder`) selected by `RN_USE_OFFLINE_EMBEDDER=1`. It never downloads
a model, so tests and CI exercise the *entire* parse→chunk→embed→upsert→idempotency
pipeline with zero network and full determinism. It is **not** for production
retrieval quality — only for fast, hermetic verification of the plumbing.

## Consequences

- No API keys, no per-call cost, no corpus text leaving the machine.
- First production run downloads the model (~130 MB for `bge-small`) and caches it;
  subsequent runs are offline. CI never pays this cost because it uses the offline
  embedder.
- `torch` + `sentence-transformers` become real dependencies (heavy, but standard).
- Vectors are reproducible for a pinned model version.
- Retrieval quality is bounded by a small open model rather than a frontier hosted
  one; revisitable in M2 by changing config and re-`reindex`-ing.

## Alternatives considered

- **Hosted embedding API** — best quality-per-effort, but adds secrets, cost, a
  network dependency in CI, and a corpus-egress concern. Rejected for M1; can be
  added later behind the same `DenseEmbedder` protocol if quality demands it.
- **Instructor / GTE / e5 large** — viable and supported via config, but a larger
  default would slow the common path with little benefit at this corpus size.
