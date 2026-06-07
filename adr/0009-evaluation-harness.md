# ADR-0009 — Multi-configuration eval harness with end-to-end metrics and a dual-mode faithfulness judge

- **Status:** Accepted
- **Date:** 2025 (M4)
- **Deciders:** project owner (intern) + reviewer
- **Relates to:** M4 (evaluation), ADR-0004 (RRF), ADR-0006 (generation), ADR-0007 (refusal), ADR-0008 (agentic routing)

## Context

The brief mandates an automated evaluation harness covering retrieval P/R @ k,
citation faithfulness (LLM-as-judge with an explicit rubric), refusal
correctness, latency (p50/p95), and approximate token cost — and requires that
the report compare **at least two** retrieval configurations. Several
sub-decisions were open:

1. **What configurations to compare?** Dense-only vs hybrid; with vs without
   filters; reranker on/off (B3 isn't implemented, so excluded).
2. **End-to-end or per-stage measurement?** Score retrieval in isolation, or
   score the doc_ids that appear in the final answer's citations?
3. **What does the judge do when no LLM is available?** Either skip the metric
   (dishonest) or fall back to a heuristic (transparent).
4. **How is "token cost" computed?** The hosted-API `usage` field isn't
   available with every backend; the offline extractive generator has none.

## Decision

- **Three configurations: `hybrid_with_filters` (baseline),
  `dense_only_with_filters` (ablate sparse), `hybrid_no_filters` (ablate
  metadata filtering).** More than the required two, because the two
  ablations isolate different design choices (ADR-0004 for sparse fusion,
  M1/M2 for server-side filtering) and the report's comparative findings
  become falsifiable.

- **End-to-end measurement.** Retrieval P/R is computed against the doc_ids
  that appear in the *final* answer's deduplicated citation block — i.e. what
  the learner actually sees. This includes the agent's filtering, route-
  specific retrieval, and the refusal gate; it's the honest answer to "is
  this system good?". Lower-level retrieval-only metrics are computable from
  the per-query JSON record if needed for drill-down.

- **Dual-mode faithfulness judge.**

  - *Heuristic mode (default).* Coverage = fraction of sentences with a `[n]`
    marker; support = fraction of markers that point at a real citation. M2
    strips invalid markers before returning, so support is structurally 1.0
    unless the M2 invariant is broken. Deterministic and hermetic; the only
    score that ever leaves CI.
  - *LLM mode (opt-in via `--llm-judge`).* The same OpenAI-compatible client
    M2 uses (ADR-0006) is pointed at a judge model with an explicit rubric:
    (1) coverage — every factual sentence has a marker; (2) support — every
    cited claim follows from the cited source, wrong-source citations count
    as unsupported. The judge returns a JSON verdict; we validate, clamp to
    [0,1], and average.

  Both modes return the **same schema**, so the report header tells the
  reader which one produced the numbers (`Judge: heuristic | llm`) and
  comparisons within one report are always like-for-like.

- **Approximate tokens via 4-chars/token.** Honest about the approximation
  (called out in the report's "Honest limitations" section), and consistent
  across both backends and all configurations. Reliable for *relative* cost
  comparison between configs, which is what the brief actually asks for.

- **Filter-ablation by monkey-patching `build_filter`.** The
  `hybrid_no_filters` configuration sets `build_filter` to a no-op for the
  duration of that configuration's run, then restores the original. Cleaner
  than threading a `use_filters` flag through every node and the entire M2
  pipeline; the eval module is the *only* caller, and the original is
  guaranteed restored before the harness returns.

- **Golden set is JSONL with strict pydantic validation.** Each line is one
  `GoldenItem`; an unexpected schema or a missing field is a loud
  `ValidationError` with a line-number prefix. The committed file has 40
  in-corpus questions (≥6 per route) plus 8 held-out out-of-corpus
  questions, of which 5 are unambiguously off-topic (router catches → out-
  of-scope refusal) and 3 are in-domain phrasings about papers not in the
  corpus (router routes them in; the M2 confidence gate refuses).

## Consequences

- `make eval` runs the whole harness offline, writes `eval/report.json` and
  `eval/report.md`, and exits 0 in CI.
- The headline table compares the three configs side-by-side; the
  per-route table makes failures actionable (e.g. low recall on
  `compare_approaches` is a different bug than low recall on `find_papers`).
- The honest-limitations section is non-optional and stays in the report
  even when the LLM judge is enabled — every approximation the harness
  makes is named where the numbers are read.
- The harness is reproducible (deterministic offline embedder + extractive
  generator + heuristic judge), so the same checkout produces the same
  report on every machine.

## Alternatives considered

- **RAGAS / TRIAD / ARES via a hosted API.** Mature batteries-included
  frameworks, but they reintroduce the secret + network dependency we've
  worked to avoid (ADR-0001 / ADR-0006), and lock the rubric to whatever the
  framework ships. Rejected as the default; the JSON judge schema is
  framework-compatible if a reviewer wants to slot RAGAS in later.
- **Score retrieval in isolation, ignore the agent layer.** Cleaner numbers
  for the retrieval system itself but blind to whether the user sees them —
  refusal gates, tag relaxation, and route-specific filters all alter what
  citations actually appear. Rejected; we measure what the learner sees.
- **Skip faithfulness when no LLM is configured.** Honest in one sense
  (don't claim a metric you can't compute) but unhelpful in CI. The dual-
  mode design lets both audiences (CI + reviewer) get the right thing.
