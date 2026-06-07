"""Orchestrate the evaluation across configurations (M4)."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from research_navigator.agents.graph import build_agent_graph
from research_navigator.agents.state import AgentRoute, AgentState
from research_navigator.config import GenerationConfig, Settings, get_settings
from research_navigator.eval.golden_set import GoldenItem, load_golden_set
from research_navigator.eval.judge import build_chunks_index, judge_answer
from research_navigator.eval.metrics import (
    approx_tokens,
    heuristic_faithfulness,
    latency_percentiles,
    precision_at_k,
    recall_at_k,
)
from research_navigator.eval.retrievers import DenseOnlyRetriever
from research_navigator.generate.pipeline import Answer, AnswerStatus, QueryEngine
from research_navigator.ingest.embed import build_embedders
from research_navigator.ingest.qdrant_store import QdrantStore
from research_navigator.logging import get_logger

log = get_logger(__name__)


class EvalConfig(BaseModel):
    """One retrieval configuration to evaluate against the golden set."""

    name: str
    dense_only: bool = False
    """If True, skip the sparse branch (ablate hybrid)."""

    drop_filters: bool = False
    """If True, do not apply Qdrant metadata filters (ablate filtering)."""


DEFAULT_CONFIGS: list[EvalConfig] = [
    EvalConfig(name="hybrid_with_filters"),
    EvalConfig(name="dense_only_with_filters", dense_only=True),
    EvalConfig(name="hybrid_no_filters", drop_filters=True),
]


class QueryResult(BaseModel):
    """Per-(query, config) result. Everything the report needs to be self-explanatory."""

    config: str
    id: str
    query: str
    route_expected: AgentRoute
    route_actual: AgentRoute | None
    expected_doc_ids: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    answer_status: str
    confidence: float
    latency_ms: float
    approx_tokens: int
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    faithfulness: dict[str, Any] = Field(default_factory=dict)
    is_out_of_corpus: bool = False
    refusal_correct: bool | None = None


class ConfigSummary(BaseModel):
    """Aggregate metrics for one configuration."""

    name: str
    n_in_corpus: int
    n_out_of_corpus: int
    precision_at_5: float
    recall_at_5: float
    faithfulness_overall: float
    refusal_correctness: float
    latency: dict[str, float]
    approx_tokens_mean: float
    per_route: dict[str, dict[str, float]]


class EvalReport(BaseModel):
    """The full structured report. Markdown rendering happens in :mod:`report`."""

    golden_set_path: str
    generator_backend: str
    judge_mode: str
    n_questions: int
    configs: list[ConfigSummary]
    results: list[QueryResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine construction per config
# ---------------------------------------------------------------------------


def _build_engine_for_config(
    settings: Settings, config: EvalConfig, store: QdrantStore | None = None
) -> QueryEngine:
    """Build a :class:`QueryEngine` with the retriever this config dictates."""
    store = store or QdrantStore(settings)
    dense, sparse = build_embedders(settings)
    if config.dense_only:
        retriever = DenseOnlyRetriever(settings, store, dense)
        return QueryEngine(settings, store=store, retriever=retriever)
    return QueryEngine(settings, store=store, dense=dense, sparse=sparse)


def _patch_filters_off(graph: Any) -> Any:
    """If ``drop_filters`` is set, the agent's filters are bypassed by monkey-patching
    :func:`research_navigator.retrieve.filters.build_filter` to always return ``None``.

    Cleaner than threading a ``use_filters`` flag through every node — this is
    what an A/B in a non-production eval is *for*.
    """
    from research_navigator.retrieve import filters as filters_mod

    filters_mod.build_filter = lambda *_a, **_k: None
    return graph


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def run_evaluation(
    golden_set_path: Path,
    *,
    settings: Settings | None = None,
    configs: Iterable[EvalConfig] | None = None,
    judge_config: GenerationConfig | None = None,
    store: QdrantStore | None = None,
) -> EvalReport:
    """Run every config against every golden-set item; return a structured report.

    Args:
        golden_set_path: Path to the JSONL golden set.
        settings: Project settings (defaults to :func:`get_settings`).
        configs: Iterable of configs to compare. Defaults to ``DEFAULT_CONFIGS``.
        judge_config: If given, use this OpenAI-compatible config for the
            LLM judge. ``None`` keeps the deterministic heuristic judge.
    """
    settings = settings or get_settings()
    configs = list(configs) if configs is not None else DEFAULT_CONFIGS
    items = load_golden_set(golden_set_path)
    log.info("eval_start", n_items=len(items), configs=[c.name for c in configs])

    # Stash + restore the real build_filter when --drop-filters is used.
    from research_navigator.retrieve import filters as filters_mod

    real_build_filter = filters_mod.build_filter

    results: list[QueryResult] = []
    summaries: list[ConfigSummary] = []

    for cfg in configs:
        log.info("eval_config_start", config=cfg.name)
        engine = _build_engine_for_config(settings, cfg, store=store)
        graph = build_agent_graph(settings, engine=engine)

        # Filter-ablation switch (per-config, restored after the loop iteration).
        if cfg.drop_filters:
            filters_mod.build_filter = lambda *_a, **_k: None
        else:
            filters_mod.build_filter = real_build_filter

        cfg_results = [_run_one_item(item, graph, judge_config, cfg.name) for item in items]
        results.extend(cfg_results)
        summaries.append(_summarise(cfg.name, cfg_results))

    # Restore the original build_filter so later code in the same process isn't affected.
    filters_mod.build_filter = real_build_filter

    return EvalReport(
        golden_set_path=str(golden_set_path),
        generator_backend=settings.generation.backend,
        judge_mode="llm" if judge_config is not None else "heuristic",
        n_questions=len(items),
        configs=summaries,
        results=results,
    )


def _run_one_item(
    item: GoldenItem,
    graph: Any,
    judge_config: GenerationConfig | None,
    config_name: str,
) -> QueryResult:
    start = time.perf_counter()
    final_dict = graph.invoke(AgentState(query=item.query))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    final = AgentState.model_validate(final_dict)
    answer: Answer | None = final.answer

    retrieved_doc_ids: list[str] = []
    if answer and answer.citations:
        seen: set[str] = set()
        for c in answer.citations:
            if c.doc_id not in seen:
                seen.add(c.doc_id)
                retrieved_doc_ids.append(c.doc_id)

    p_at_5 = precision_at_k(retrieved_doc_ids, item.expected_doc_ids, k=5)
    r_at_5 = recall_at_k(retrieved_doc_ids, item.expected_doc_ids, k=5)

    if answer is None:
        faithfulness: dict[str, Any] = {"coverage": 0.0, "support": 0.0, "overall": 0.0}
    elif item.is_out_of_corpus or answer.status is not AnswerStatus.ANSWERED:
        faithfulness = {"coverage": 0.0, "support": 0.0, "overall": 0.0, "judge": "n/a"}
    else:
        if judge_config is not None:
            faithfulness = judge_answer(
                item.query, answer, build_chunks_index(), config=judge_config
            )
        else:
            faithfulness = {**heuristic_faithfulness(answer), "judge": "heuristic"}

    refusal_correct: bool | None = None
    if item.expected_refusal:
        refusal_correct = answer is not None and answer.status is not AnswerStatus.ANSWERED

    return QueryResult(
        config=config_name,
        id=item.id,
        query=item.query,
        route_expected=item.route,
        route_actual=final.route,
        expected_doc_ids=list(item.expected_doc_ids),
        retrieved_doc_ids=retrieved_doc_ids,
        answer_status=answer.status.value if answer else "missing",
        confidence=answer.confidence if answer else 0.0,
        latency_ms=round(elapsed_ms, 2),
        approx_tokens=approx_tokens(item.query, answer.text if answer else ""),
        precision_at_5=round(p_at_5, 4),
        recall_at_5=round(r_at_5, 4),
        faithfulness=faithfulness,
        is_out_of_corpus=item.is_out_of_corpus,
        refusal_correct=refusal_correct,
    )


def _summarise(name: str, results: list[QueryResult]) -> ConfigSummary:
    in_corpus = [r for r in results if not r.is_out_of_corpus]
    ooc = [r for r in results if r.is_out_of_corpus]

    def _mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    per_route: dict[str, dict[str, float]] = defaultdict(dict)
    for route in AgentRoute:
        bucket = [r for r in in_corpus if r.route_expected is route]
        if not bucket:
            continue
        per_route[route.value] = {
            "n": float(len(bucket)),
            "precision_at_5": _mean([r.precision_at_5 for r in bucket]),
            "recall_at_5": _mean([r.recall_at_5 for r in bucket]),
            "faithfulness_overall": _mean(
                [float(r.faithfulness.get("overall", 0.0)) for r in bucket]
            ),
        }

    refusal_scores = [r for r in results if r.refusal_correct is not None]
    refusal_pct = (
        sum(1 for r in refusal_scores if r.refusal_correct) / len(refusal_scores)
        if refusal_scores
        else 1.0
    )

    return ConfigSummary(
        name=name,
        n_in_corpus=len(in_corpus),
        n_out_of_corpus=len(ooc),
        precision_at_5=_mean([r.precision_at_5 for r in in_corpus]),
        recall_at_5=_mean([r.recall_at_5 for r in in_corpus]),
        faithfulness_overall=_mean([float(r.faithfulness.get("overall", 0.0)) for r in in_corpus]),
        refusal_correctness=round(refusal_pct, 4),
        latency=latency_percentiles([r.latency_ms for r in results]),
        approx_tokens_mean=_mean([float(r.approx_tokens) for r in results]),
        per_route=dict(per_route),
    )
