# ADR-0007 — Refusal on a dense-cosine confidence signal, not the fused RRF score

- **Status:** Accepted
- **Date:** 2025 (M2)
- **Deciders:** project owner (intern) + reviewer
- **Relates to:** M2 (refusal path), ADR-0004 (RRF fusion), ADR-0006 (generation)

## Context

The brief requires a **low-confidence refusal**: if top-k retrieval similarity is
below a tuned threshold, return the fixed "I don't have enough relevant material…"
message rather than fabricate an answer. This is "Decisions to lock" #5 (refusal
threshold — how it's tuned and on what signal). Two questions had to be answered:
**which signal** measures confidence, and **what threshold** on it.

The obvious candidate — the score the retriever already returns — is the wrong one.
Our hybrid retriever fuses dense and sparse branches with **Reciprocal Rank Fusion**
(ADR-0004). RRF scores are computed from *ranks*, not similarities: a top hit scores
around `1/(k+1)` (≈ `0.016` at the default `k=60`), and the value reflects *agreement
between the two rankings*, not *how semantically close* the best chunk is to the
query. A nonsense query still produces a #1-ranked chunk and therefore a perfectly
ordinary-looking RRF score. Thresholding on RRF would refuse almost nothing.

## Decision

**Confidence = the top dense cosine similarity among the filtered candidates**, taken
from a **separate dense-only `query_points` pass** that carries the *same* metadata
filter as the fused query. The retriever issues this second query specifically to get
a similarity-calibrated number; it returns it as `RetrievalResult.dense_confidence`.

The pipeline refuses when **`dense_confidence < RN_RETRIEVAL__REFUSAL_MIN_SCORE`**
(default **`0.30`**) *or* when retrieval returned nothing at all (e.g. a metadata
filter that no document satisfies). Cosine is bounded `[-1, 1]` and, for a normalised
sentence-embedding model like `bge-small` (ADR-0001), a genuinely on-topic top chunk
sits well above `0.3` while out-of-corpus questions fall below it; `0.30` is the
conservative floor in that gap and is exposed as config so it can be re-tuned against
the M4 held-out out-of-corpus set without touching code.

Refusal is also enforced **after** generation, not only before it: markers an answer
emits are validated against the real citation set, and an answer left with no valid
attribution becomes a refusal (ADR-0006). So "low confidence" and "could not attribute"
both resolve to the same graceful decline.

## Consequences

- The refusal gate measures the right thing (semantic closeness), so it actually
  fires on out-of-corpus questions instead of waving them through.
- One extra Qdrant query per question (dense-only, `limit=1`, no payload). Cheap, and
  it keeps the confidence signal independent of the fusion math, which can change
  (ADR-0004 leaves room for weighted fusion) without disturbing the refusal logic.
- The threshold is model-dependent. It is correct for the production `bge-small`
  default and is config, not a constant; M4 will tune it on real refusal-correctness
  data and the number may move.
- **Test caveat (documented honestly):** the offline hashing embedder (ADR-0001) has
  no semantic geometry — its cosine is dominated by hash collisions, so an
  out-of-corpus query can spuriously clear *any* fixed threshold. The refusal unit
  tests therefore do **not** lean on embedding quality; they force the two real code
  paths deterministically: an impossibly high threshold (confidence below it) and a
  filter that matches no document (empty result). With the real model, the `0.30`
  floor does the work. This is a property of the *test embedder*, not of the gate.

## Alternatives considered

- **Threshold the fused RRF score** — simplest (no second query) but measures rank
  agreement, not similarity, so it cannot distinguish on-topic from off-topic. The
  reason this ADR exists. Rejected.
- **Min-max normalise RRF into `[0,1]` and threshold that** — still rank-derived;
  normalisation changes the range, not the meaning. Rejected.
- **A learned/LLM "is this answerable from these chunks?" judge** — more robust, but
  adds a model call (and possibly a key) to every query just to decide whether to
  proceed, and is itself fallible. Rejected for M2; the cosine gate plus the
  post-generation attribution check covers the acceptance criterion. An LLM judge is
  reserved for *evaluating* faithfulness in M4, not for gating live queries.
- **Refuse only on empty results (no similarity floor)** — would answer confidently
  from weakly-related chunks whenever *something* passes the filter. Rejected; that
  is exactly the fabrication failure mode the brief warns against.
