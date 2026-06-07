"""Scoring helpers for the eval harness (M4).

Pure, deterministic, dependency-free. Every function is independently unit-tested.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from research_navigator.generate.pipeline import Answer, AnswerStatus

# Approximate-tokens conversion. The exact tokenizer depends on the LLM and
# isn't accessible offline, so we use the widely-cited "4 chars ≈ 1 token"
# heuristic. Documented in ADR-0009; honest enough for relative comparisons.
CHARS_PER_TOKEN = 4

_MARKER = re.compile(r"\[\d+\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def precision_at_k(retrieved_doc_ids: list[str], expected: Iterable[str], k: int = 5) -> float:
    """|retrieved ∩ expected| / |retrieved|, computed over the top-K (deduped)."""
    expected_set = set(expected)
    seen: list[str] = []
    for d in retrieved_doc_ids:
        if d not in seen:
            seen.append(d)
        if len(seen) >= k:
            break
    if not seen:
        return 0.0
    return sum(1 for d in seen if d in expected_set) / len(seen)


def recall_at_k(retrieved_doc_ids: list[str], expected: Iterable[str], k: int = 5) -> float:
    """|retrieved ∩ expected| / |expected|, computed over the top-K (deduped)."""
    expected_set = set(expected)
    if not expected_set:
        return 1.0  # nothing to recall — vacuously perfect
    seen: list[str] = []
    for d in retrieved_doc_ids:
        if d not in seen:
            seen.append(d)
        if len(seen) >= k:
            break
    return sum(1 for d in seen if d in expected_set) / len(expected_set)


def latency_percentiles(timings_ms: list[float]) -> dict[str, float]:
    """Return p50, p95, mean, max for a list of millisecond timings."""
    if not timings_ms:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0}
    sorted_t = sorted(timings_ms)

    def pct(p: float) -> float:
        # Linear-interpolation percentile (NumPy "linear" / NIST 7).
        if len(sorted_t) == 1:
            return sorted_t[0]
        idx = (len(sorted_t) - 1) * p
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return sorted_t[lo]
        return sorted_t[lo] + (sorted_t[hi] - sorted_t[lo]) * (idx - lo)

    return {
        "p50": round(pct(0.50), 2),
        "p95": round(pct(0.95), 2),
        "mean": round(sum(sorted_t) / len(sorted_t), 2),
        "max": round(sorted_t[-1], 2),
    }


def approx_tokens(*texts: str) -> int:
    """Approximate-token count for one or more pieces of text (4 chars / token)."""
    total_chars = sum(len(t) for t in texts)
    return max(1, total_chars // CHARS_PER_TOKEN)


def heuristic_faithfulness(answer: Answer) -> dict[str, float]:
    """Deterministic, hermetic faithfulness score (no LLM needed).

    Two sub-scores, both in [0.0, 1.0]:

      * **coverage** — fraction of sentences in the answer that carry a
        ``[n]`` citation marker. A perfectly cited answer scores 1.0.
      * **support** — fraction of cited markers that point to a real citation
        (i.e. survive the M2 marker-validation step). 1.0 by construction in
        the M2 pipeline; included here so the LLM judge can lower it.

    The brief asks for an LLM-as-judge, but a hermetic baseline is essential
    for CI and for the report's reproducibility section. The LLM judge in
    :mod:`research_navigator.eval.judge` returns the same schema so the two
    are drop-in interchangeable.
    """
    if answer.status is not AnswerStatus.ANSWERED or not answer.text.strip():
        return {"coverage": 0.0, "support": 0.0, "overall": 0.0}

    sentences = [s for s in _SENT_SPLIT.split(answer.text.strip()) if s.strip()]
    if not sentences:
        return {"coverage": 0.0, "support": 0.0, "overall": 0.0}

    with_marker = sum(1 for s in sentences if _MARKER.search(s))
    coverage = with_marker / len(sentences)

    # Every marker in answer.text must point at a real citation — the M2
    # pipeline strips out-of-range markers before returning, so this is 1.0
    # unless the pipeline's invariant was broken.
    valid_numbers = {c.number for c in answer.citations}
    markers = [int(m) for m in re.findall(r"\[(\d+)\]", answer.text)]
    support = sum(1 for n in markers if n in valid_numbers) / len(markers) if markers else 0.0
    overall = round((coverage + support) / 2, 4)
    return {
        "coverage": round(coverage, 4),
        "support": round(support, 4),
        "overall": overall,
    }
