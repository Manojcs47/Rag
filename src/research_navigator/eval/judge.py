"""LLM-as-judge for citation faithfulness (M4).

A second model reads the question, the answer (with ``[n]`` markers), and the
numbered sources, then returns a structured verdict against an **explicit
rubric** (defined below). Mirrors the OpenAI-compatible client pattern from
:mod:`research_navigator.generate.generator` so no new secret/transport is
introduced. Falls back to the heuristic scorer when no LLM is configured.

Rubric — the judge answers two yes/no/partial questions, scored in [0.0, 1.0]:

  1. **Coverage.** Does every factual sentence carry at least one ``[n]``
     citation marker? (Opinion/transition sentences without a marker are
     fine; missing citations on factual claims are not.)
  2. **Support.** For every cited claim, does the cited source actually
     support that claim? Wrong citations (citing source [2] for a claim
     that's only in [3]) count as unsupported.

The final ``overall`` is the mean of the two; this is the number that appears
in the headline report table.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from research_navigator.config import GenerationConfig
from research_navigator.eval.metrics import heuristic_faithfulness
from research_navigator.generate.pipeline import Answer, AnswerStatus
from research_navigator.logging import get_logger
from research_navigator.retrieve.hybrid import RetrievedChunk

log = get_logger(__name__)


JUDGE_SYSTEM_PROMPT = (
    "You are an evaluator scoring the citation faithfulness of an AI assistant's "
    "answer. Read the QUESTION, the ANSWER (which contains [n] citation markers), "
    "and the numbered SOURCES, then score on two axes:\n\n"
    "1. COVERAGE — does every factual sentence in the ANSWER carry at least one "
    "[n] marker? (Transition or opinion sentences do not need one.)\n"
    "2. SUPPORT — for every cited claim, does the source actually support that "
    "claim? Citing the wrong source counts as unsupported.\n\n"
    "Respond with ONLY a JSON object of the form:\n"
    '{"coverage": 0.0, "support": 0.0, "issues": ["..."], "reasoning": "..."}\n'
    "Each score is a float in [0.0, 1.0]. Do not include any other text."
)


def judge_answer(
    query: str,
    answer: Answer,
    chunks_by_doc_id: dict[str, RetrievedChunk],
    *,
    config: GenerationConfig | None = None,
) -> dict[str, Any]:
    """Score one answer's citation faithfulness.

    Falls back to :func:`heuristic_faithfulness` when ``config`` is ``None`` or
    the LLM call fails, so the harness is always callable.
    """
    if answer.status is not AnswerStatus.ANSWERED or not answer.citations:
        return {**heuristic_faithfulness(answer), "judge": "heuristic", "reason": "no_answer"}

    if config is None:
        return {**heuristic_faithfulness(answer), "judge": "heuristic"}

    user_prompt = _build_judge_prompt(query, answer, chunks_by_doc_id)
    headers = {"Content-Type": "application/json"}
    if config.llm_api_key is not None:
        headers["Authorization"] = f"Bearer {config.llm_api_key.get_secret_value()}"

    try:
        resp = httpx.post(
            f"{config.llm_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": config.llm_model,
                "temperature": 0.0,
                "max_tokens": 400,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=config.llm_timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        verdict = _parse_verdict(text)
        verdict["judge"] = "llm"
        verdict["overall"] = round((verdict["coverage"] + verdict["support"]) / 2, 4)
        log.info("judge_done", overall=verdict["overall"])
        return verdict
    except Exception as exc:  # network error, bad JSON, etc.
        log.warning("judge_failed_fallback", error=str(exc))
        h = heuristic_faithfulness(answer)
        return {**h, "judge": "heuristic", "reason": f"llm_failed: {exc}"}


def _build_judge_prompt(
    query: str, answer: Answer, chunks_by_doc_id: dict[str, RetrievedChunk]
) -> str:
    blocks = []
    for c in answer.citations:
        chunk = chunks_by_doc_id.get(c.doc_id)
        excerpt = chunk.text.strip() if chunk else ""
        blocks.append(f"[{c.number}] {c.title} ({c.source}, {c.year})\n{excerpt}")
    sources_blob = "\n\n".join(blocks)
    return (
        f"QUESTION: {query}\n\n"
        f"ANSWER:\n{answer.text}\n\n"
        f"SOURCES:\n{sources_blob}\n\n"
        "Return the JSON verdict now."
    )


def _parse_verdict(text: str) -> dict[str, Any]:
    """Extract the JSON object from a model response, tolerating prose around it."""
    # Find the first JSON-looking braces.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in judge response: {text!r}")
    parsed = json.loads(m.group(0))
    coverage = float(parsed.get("coverage", 0.0))
    support = float(parsed.get("support", 0.0))
    return {
        "coverage": round(max(0.0, min(1.0, coverage)), 4),
        "support": round(max(0.0, min(1.0, support)), 4),
        "issues": list(parsed.get("issues", []) or []),
        "reasoning": str(parsed.get("reasoning", "") or ""),
    }


def build_chunks_index(
    *,
    chunks_by_doc_id: dict[str, RetrievedChunk] | None = None,
    fallback_chunks: list[RetrievedChunk] | None = None,
) -> dict[str, RetrievedChunk]:
    """Build a doc_id → best-scoring chunk index used by the judge for excerpts."""
    if chunks_by_doc_id is not None:
        return chunks_by_doc_id
    out: dict[str, RetrievedChunk] = {}
    for chunk in fallback_chunks or []:
        existing = out.get(chunk.doc_id)
        if existing is None or chunk.score > existing.score:
            out[chunk.doc_id] = chunk
    return out
