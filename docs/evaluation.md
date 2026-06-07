# Evaluation harness (M4)

`make eval` runs the golden set across every configuration, writes a JSON dump
of every per-query verdict, and renders a one-page Markdown summary. The
harness is hermetic: with the default backend (`extractive` + heuristic judge)
it needs only the local Qdrant and the deterministic offline embedder.

## Configurations compared

| ID | Name | Sparse | Filters |
|----|------|--------|---------|
| A | `hybrid_with_filters` | yes | yes |
| B | `dense_only_with_filters` | no | yes |
| C | `hybrid_no_filters` | yes | no |

A vs B isolates the contribution of the sparse branch (ADR-0004 fusion).
A vs C isolates the contribution of server-side metadata filtering (M1/M2).

## Metrics

- **Retrieval P/R @ 5.** Computed on the deduplicated doc_ids in the final
  answer's citation block, against the golden-set `expected_doc_ids`.
  "End-to-end" by design — see ADR-0009.
- **Citation faithfulness.** Heuristic by default (coverage + support over
  `[n]` markers, both in [0,1], averaged). LLM judge via `--llm-judge` runs
  the same rubric through an OpenAI-compatible model.
- **Refusal correctness.** Fraction of out-of-corpus held-out queries on
  which the system refuses (any non-`ANSWERED` status counts).
- **Latency.** Per-query wall-clock time → p50, p95, mean, max across the
  golden set.
- **Approximate tokens.** `(prompt_chars + answer_chars) / 4` per query.
  Reliable for cross-config comparison; not for billing.

## Running

```bash
# 1. Make sure the corpus is ingested into Qdrant.
make up
uv run python -m research_navigator.ingest ingest

# 2. Run the eval (writes eval/report.json + eval/report.md).
make eval

# 3. View the one-page report.
cat eval/report.md
```

To use an LLM judge instead of the heuristic:

```bash
# Make sure RN_GENERATION__LLM_BASE_URL/MODEL/API_KEY point at a working model.
uv run python -m research_navigator.cli.eval --llm-judge
```

## Re-tuning the refusal threshold

The eval is the canonical place to re-tune `RN_RETRIEVAL__REFUSAL_MIN_SCORE`
(ADR-0007). Inspect `refusal_correctness` and the false-refusal rate in the
in-corpus per-route table, sweep a few thresholds, and pick the value with
the best refusal–coverage tradeoff. Default 0.30 is the conservative floor
calibrated for `bge-small`; expect to land slightly lower on this corpus.
