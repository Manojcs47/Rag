"""Render an :class:`EvalReport` to JSON + Markdown (M4)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from research_navigator.eval.runner import ConfigSummary, EvalReport


def write_report(report: EvalReport, out_dir: Path) -> tuple[Path, Path]:
    """Write ``report.json`` and ``report.md`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _render_markdown(report: EvalReport) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts: list[str] = [
        "# AI Research Navigator — Evaluation Report",
        "",
        f"_Generated: {now}_",
        f"_Generator backend: `{report.generator_backend}` · "
        f"Judge: `{report.judge_mode}` · "
        f"Golden set: `{report.golden_set_path}` ({report.n_questions} questions)_",
        "",
        "## Configurations",
        "",
        "| Name | Sparse branch | Metadata filters |",
        "|---|---|---|",
    ]
    for cfg in report.configs:
        sparse = "no" if cfg.name == "dense_only_with_filters" else "yes"
        filters = "no" if cfg.name == "hybrid_no_filters" else "yes"
        parts.append(f"| `{cfg.name}` | {sparse} | {filters} |")

    parts += ["", "## Headline metrics", "", _headline_table(report.configs), ""]
    parts += ["## Latency & approximate cost", "", _latency_table(report.configs), ""]
    parts += ["## Per-route breakdown", "", _per_route_table(report.configs), ""]
    parts += ["## Findings", "", _findings(report.configs), ""]
    parts += [
        "## Honest limitations",
        "",
        "- **Approximate tokens.** Counts are derived from a 4-chars-per-token "
        "heuristic, not the LLM's true tokeniser; they are reliable only for "
        "cross-config *comparison*, not for billing estimates.",
        "- **Heuristic faithfulness in offline mode.** When the report header reads "
        "`Judge: heuristic`, the score reflects citation-marker coverage and "
        "marker validity only — it cannot detect a syntactically-correct citation "
        "that is semantically wrong. Re-run with `--llm-judge` (set "
        "`RN_GENERATION__BACKEND=openai` and point `RN_GENERATION__LLM_*` at a "
        "running model) for a semantic judgement.",
        "- **Golden-set labelling.** Expected doc_ids are author-annotated, not "
        "exhaustively verified; missing relevant docs at the labelling step "
        "lower the measured recall artificially.",
        "- **Refusal sample size.** The out-of-corpus held-out set is small "
        f"(n={report.configs[0].n_out_of_corpus if report.configs else 0}); "
        "treat refusal-correctness deltas with appropriate scepticism.",
        "",
    ]
    return "\n".join(parts) + "\n"


def _headline_table(configs: list[ConfigSummary]) -> str:
    header = "| Metric | " + " | ".join(f"`{c.name}`" for c in configs) + " |"
    sep = "|---|" + "|".join("---" for _ in configs) + "|"

    def row(label: str, values: list[float], fmt: str = "{:.3f}") -> str:
        return "| " + label + " | " + " | ".join(fmt.format(v) for v in values) + " |"

    return "\n".join(
        [
            header,
            sep,
            row("Retrieval P@5", [c.precision_at_5 for c in configs]),
            row("Retrieval R@5", [c.recall_at_5 for c in configs]),
            row("Citation faithfulness", [c.faithfulness_overall for c in configs]),
            row("Refusal correctness", [c.refusal_correctness for c in configs]),
        ]
    )


def _latency_table(configs: list[ConfigSummary]) -> str:
    header = "| Metric | " + " | ".join(f"`{c.name}`" for c in configs) + " |"
    sep = "|---|" + "|".join("---" for _ in configs) + "|"

    def row(label: str, values: list[float], fmt: str = "{:.1f}") -> str:
        return "| " + label + " | " + " | ".join(fmt.format(v) for v in values) + " |"

    return "\n".join(
        [
            header,
            sep,
            row("Latency p50 (ms)", [c.latency["p50"] for c in configs]),
            row("Latency p95 (ms)", [c.latency["p95"] for c in configs]),
            row("Mean approx tokens/query", [c.approx_tokens_mean for c in configs]),
        ]
    )


def _per_route_table(configs: list[ConfigSummary]) -> str:
    routes: list[str] = []
    seen: set[str] = set()
    for cfg in configs:
        for r in cfg.per_route:
            if r not in seen:
                seen.add(r)
                routes.append(r)
    header = "| Route | n | " + " | ".join(f"P@5 ({c.name})" for c in configs) + " |"
    sep = "|---|---|" + "|".join("---" for _ in configs) + "|"
    rows = [header, sep]
    for route in routes:
        n_val = next(
            (cfg.per_route[route]["n"] for cfg in configs if route in cfg.per_route),
            0.0,
        )
        cells = [
            f"{cfg.per_route.get(route, {}).get('precision_at_5', 0.0):.3f}" for cfg in configs
        ]
        rows.append(f"| {route} | {int(n_val)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _findings(configs: list[ConfigSummary]) -> str:
    if len(configs) < 2:
        return "_Need at least two configurations for comparative findings._"
    base = configs[0]
    bullets = []
    for other in configs[1:]:
        d_p = base.precision_at_5 - other.precision_at_5
        d_r = base.recall_at_5 - other.recall_at_5
        d_ref = base.refusal_correctness - other.refusal_correctness
        bullets.append(
            f"- **`{base.name}` vs `{other.name}`:** "
            f"ΔP@5 = {d_p:+.3f}, ΔR@5 = {d_r:+.3f}, "
            f"Δrefusal = {d_ref:+.3f}, "
            f"Δlatency p50 = {base.latency['p50'] - other.latency['p50']:+.1f} ms."
        )
    return "\n".join(bullets)
