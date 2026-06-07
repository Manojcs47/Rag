# Architecture Decision Records

Each ADR captures one significant, hard-to-reverse decision: the context that
forced it, the choice made, the consequences accepted, and the alternatives
rejected. They are immutable once `Accepted` — a later decision that overrides an
earlier one is a *new* ADR that marks the old one `Superseded`.

Format is lightweight [MADR](https://adr.github.io/madr/): Status · Context ·
Decision · Consequences · Alternatives.

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-dense-embedding-model.md) | Local open-source dense embeddings via sentence-transformers | Accepted |
| [0002](0002-package-and-env-management-uv.md) | uv for packaging and environment management | Accepted |
| [0003](0003-chunking-strategy.md) | Section-bounded, content-type-aware chunking | Accepted |
| [0004](0004-sparse-retrieval-and-fusion.md) | TF sparse vectors + Qdrant IDF (BM25) with RRF fusion | Accepted |
| [0005](0005-deterministic-point-ids-and-idempotency.md) | Deterministic UUIDv5 point IDs for idempotent ingestion | Accepted |
| [0006](0006-generation-backend-and-generator-abstraction.md) | Extractive-by-default generation behind an OpenAI-compatible backend | Accepted |
| [0007](0007-refusal-threshold-and-confidence-signal.md) | Refusal on a dense-cosine confidence signal, not the fused RRF score | Accepted |
