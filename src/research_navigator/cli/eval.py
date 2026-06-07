"""CLI for the evaluation harness (M4).

    python -m research_navigator.cli.eval                        # default golden set + configs
    python -m research_navigator.cli.eval --golden eval/my.jsonl --out eval/
    python -m research_navigator.cli.eval --llm-judge            # use the LLM judge

Thin by design: parse args, run the harness, write report files, exit 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from research_navigator.config import get_settings
from research_navigator.eval.report import write_report
from research_navigator.eval.runner import run_evaluation
from research_navigator.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research_navigator.cli.eval")
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("eval/golden_set.jsonl"),
        help="Path to the golden-set JSONL (default: eval/golden_set.jsonl).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval"),
        help="Directory to write report.json and report.md (default: eval/).",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use the configured OpenAI-compatible model as the faithfulness judge.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    judge_config = settings.generation if args.llm_judge else None
    report = run_evaluation(args.golden, settings=settings, judge_config=judge_config)
    json_path, md_path = write_report(report, args.out)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
