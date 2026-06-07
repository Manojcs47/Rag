# ADR-0005 — Deterministic UUIDv5 point IDs for idempotent ingestion

- **Status:** Accepted
- **Date:** 2025 (M1)
- **Deciders:** project owner (intern) + reviewer
- **Relates to:** B3 (storage), B4 (idempotency), ADR-0003 (chunking)

## Context

Ingestion must be **idempotent** (B4): re-running it over an unchanged corpus must
write nothing, and changing one document must touch only that document's chunks —
not trigger a full re-embed of everything. That requires a chunk's storage id to be
a pure function of its *identity and content*, computed the same way on every run.

Two constraints collided during M1:

1. We want an id derived from `(doc_id, chunk_index, content_hash)` so that identical
   chunks map to identical ids and changed chunks map to new ids.
2. **Qdrant only accepts point ids that are an unsigned integer or a UUID.** A raw
   SHA-1/SHA-256 hex digest is neither, and Qdrant rejects it. (This was found by
   testing against a real Qdrant, not assumed.)

## Decision

Derive each point id as a **UUIDv5** over a fixed namespace from the chunk identity
string:

```
point_id = uuid5(NAMESPACE, f"{doc_id}:{chunk_index}:{content_hash}")
```

where `content_hash = sha256(text)[:16]` and `NAMESPACE` is a hard-coded constant so
the mapping is stable across runs and machines. UUIDv5 is deterministic (unlike v4),
Qdrant-valid, and collision-resistant for our inputs.

Ingestion then diffs per document: compute the desired id set, fetch the existing id
set for that `doc_id` from Qdrant, **upsert only the new ids, delete only the stale
ones, and skip entirely when the two sets match**. Errors are reported per document,
never swallowed; a bad document is logged and surfaced in the run report while the
rest proceed.

## Consequences

- Re-ingesting an unchanged corpus is a no-op (verified end-to-end: 3387 chunks → 0
  writes on the second run).
- Editing one document re-embeds and upserts only its changed chunks and removes only
  its now-orphaned chunks; the rest of the index is untouched (covered by an
  integration test).
- Embedding cost on routine re-runs drops to near zero, which matters under ADR-0001
  (local model, but still CPU time).
- `content_hash` is truncated to 16 hex chars (64 bits); collision probability is
  negligible at this corpus size. Full-length hashing is a one-line change if ever
  needed.
- The namespace constant is part of the contract: changing it would re-id every
  point and force a full reindex. It is therefore fixed and documented here.

## Alternatives considered

- **Raw SHA hex as the id** — the obvious first attempt; **rejected because Qdrant
  rejects non-UUID/non-int ids.**
- **Random UUIDv4 per point** — trivially unique but non-deterministic, which breaks
  idempotency entirely. Rejected.
- **Auto-increment / sequential ids** — would require an external id↔chunk mapping
  store and careful deletion bookkeeping. Rejected; the content-derived UUIDv5 needs
  no side table.
