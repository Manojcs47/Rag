# ADR-0004 — TF sparse vectors + Qdrant IDF (BM25) with RRF fusion

- **Status:** Accepted
- **Date:** 2025 (M1, consumed by M2 retrieval)
- **Deciders:** project owner (intern) + reviewer
- **Relates to:** ADR-0001 (dense), B2 (embedding), B3 (storage)

## Context

Dense embeddings retrieve on meaning but miss exact lexical matches — author names,
acronyms, method names, rare tokens. A hybrid (dense + sparse) retriever is the
standard fix. The sparse side can be a learned model (SPLADE) or a classical lexical
scheme (BM25). We want lexical recall **without** a second model download and with
full determinism, consistent with ADR-0001.

Qdrant supports named sparse vectors and can apply an **IDF modifier** server-side,
which means a client only needs to supply term-frequency vectors; Qdrant turns them
into BM25-style scores.

## Decision

- **Sparse vectors are log-scaled term frequencies** over a hashed vocabulary
  (`HashingSparseEmbedder`): `value = 1 + ln(tf)` per term bucket. No model, no
  vocabulary file, fully deterministic.
- **IDF is supplied by Qdrant** via `models.Modifier.IDF` on the sparse vector
  config. TF (client) × IDF (server) ≈ **BM25**, without us shipping a corpus IDF
  table.
- **Fusion is Reciprocal Rank Fusion (RRF)**, which Qdrant supports natively in its
  query API. RRF is rank-based, so it needs no score normalisation between the dense
  (cosine) and sparse (BM25) spaces — a known failure mode of weighted-sum fusion.
- Vocabulary size (`RN_SPARSE_VOCAB_SIZE`, default 2^18) and the named-vector keys
  are configurable.

(The dense + sparse vectors are stored together per point in M1/B3; the actual
fused query lands in M2. This ADR fixes the storage + scoring scheme so M2 builds on
a stable foundation.)

## Consequences

- Lexical recall with zero extra model downloads and deterministic vectors.
- BM25 scoring lives where the data lives (Qdrant), so it stays correct as the
  collection grows; the client stays dumb.
- Hashing collisions are possible at 2^18 buckets but negligible at this corpus
  size; vocab size is tunable if needed.
- A learned sparse model (SPLADE) could outperform BM25 on some queries; we forgo
  that for now. The `SparseEmbedder` protocol leaves room to swap one in later.
- RRF ignores score magnitudes; if M2 finds magnitude matters for some queries, a
  weighted fusion can be added behind the same query layer.

## Alternatives considered

- **SPLADE (learned sparse)** — strong, but a second model download, heavier, and
  non-deterministic w.r.t. version. Rejected for M1 on the same grounds as the
  hosted dense API.
- **Client-side BM25 (rank-bm25)** — would require maintaining a corpus-wide IDF
  table in the client and keeping it in sync with Qdrant on every change. Rejected;
  Qdrant's IDF modifier removes the bookkeeping.
- **Weighted-sum fusion** of normalised dense+sparse scores — sensitive to
  normalisation choices across incomparable score spaces. Rejected in favour of RRF.
