# AI Research Navigator — Evaluation Report

_Generated: 2026-06-07 16:32 UTC_
_Generator backend: `openai` · Judge: `llm` · Golden set: `eval/golden_set.jsonl` (48 questions)_

## Configurations

| Name | Sparse branch | Metadata filters |
|---|---|---|
| `hybrid_with_filters` | yes | yes |
| `dense_only_with_filters` | no | yes |
| `hybrid_no_filters` | yes | no |

## Headline metrics

| Metric | `hybrid_with_filters` | `dense_only_with_filters` | `hybrid_no_filters` |
|---|---|---|---|
| Retrieval P@5 | 0.133 | 0.133 | 0.133 |
| Retrieval R@5 | 0.142 | 0.142 | 0.142 |
| Citation faithfulness | 0.194 | 0.194 | 0.194 |
| Refusal correctness | 1.000 | 1.000 | 1.000 |

## Latency & approximate cost

| Metric | `hybrid_with_filters` | `dense_only_with_filters` | `hybrid_no_filters` |
|---|---|---|---|
| Latency p50 (ms) | 84.5 | 53.9 | 69.5 |
| Latency p95 (ms) | 184.6 | 111.8 | 164.1 |
| Mean approx tokens/query | 38.2 | 38.2 | 38.2 |

## Per-route breakdown

| Route | n | P@5 (hybrid_with_filters) | P@5 (dense_only_with_filters) | P@5 (hybrid_no_filters) |
|---|---|---|---|---|
| concept_explanation | 12 | 0.000 | 0.000 | 0.000 |
| paper_deep_dive | 7 | 0.000 | 0.000 | 0.000 |
| compare_approaches | 6 | 0.000 | 0.000 | 0.000 |
| recent_developments | 7 | 0.000 | 0.000 | 0.000 |
| find_papers | 8 | 0.667 | 0.667 | 0.667 |

## Findings

- **`hybrid_with_filters` vs `dense_only_with_filters`:** ΔP@5 = +0.000, ΔR@5 = +0.000, Δrefusal = +0.000, Δlatency p50 = +30.6 ms.
- **`hybrid_with_filters` vs `hybrid_no_filters`:** ΔP@5 = +0.000, ΔR@5 = +0.000, Δrefusal = +0.000, Δlatency p50 = +15.0 ms.

## Honest limitations

- **Approximate tokens.** Counts are derived from a 4-chars-per-token heuristic, not the LLM's true tokeniser; they are reliable only for cross-config *comparison*, not for billing estimates.
- **Heuristic faithfulness in offline mode.** When the report header reads `Judge: heuristic`, the score reflects citation-marker coverage and marker validity only — it cannot detect a syntactically-correct citation that is semantically wrong. Re-run with `--llm-judge` (set `RN_GENERATION__BACKEND=openai` and point `RN_GENERATION__LLM_*` at a running model) for a semantic judgement.
- **Golden-set labelling.** Expected doc_ids are author-annotated, not exhaustively verified; missing relevant docs at the labelling step lower the measured recall artificially.
- **Refusal sample size.** The out-of-corpus held-out set is small (n=8); treat refusal-correctness deltas with appropriate scepticism.
