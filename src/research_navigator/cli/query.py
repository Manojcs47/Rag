"""CLI for the M2 query pipeline.

    python -m research_navigator.cli.query "What is RAG?"
    python -m research_navigator.cli.query "recent open-weight models" --json

Thin by design: it parses arguments, builds a :class:`QueryEngine`, runs one query,
and renders the result. All business logic lives in ``retrieve``/``generate``.
Exit code is non-zero only on a backend error — a refusal is a valid outcome.
"""

from __future__ import annotations

import argparse
import json
import sys

from research_navigator.config import get_settings
from research_navigator.generate.pipeline import Answer, AnswerStatus, QueryEngine
from research_navigator.logging import configure_logging


def _answer_dict(answer: Answer) -> dict[str, object]:
    return {
        "status": answer.status.value,
        "query": answer.query,
        "text": answer.text,
        "confidence": round(answer.confidence, 4),
        "num_retrieved": answer.num_retrieved,
        "citations": [c.model_dump() for c in answer.citations],
        "error": answer.error,
    }


def main(argv: list[str] | None = None) -> int:
    """Entrypoint. Returns a process exit code (non-zero only on backend error)."""
    parser = argparse.ArgumentParser(prog="research_navigator.cli.query")
    parser.add_argument("query", help="The question to answer.")
    parser.add_argument(
        "--json", action="store_true", help="Emit the full answer as JSON instead of Markdown."
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    answer = QueryEngine(settings).answer(args.query)

    if args.json:
        print(json.dumps(_answer_dict(answer), indent=2, ensure_ascii=False))
    else:
        print(answer.render())

    return 1 if answer.status is AnswerStatus.ERROR else 0


if __name__ == "__main__":
    sys.exit(main())
