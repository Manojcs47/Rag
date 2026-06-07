"""CLI for the ingestion pipeline (B6).

    python -m research_navigator.ingest ingest [--doc DOC_ID] [--force]
    python -m research_navigator.ingest validate
    python -m research_navigator.ingest reindex
    python -m research_navigator.ingest stats

This module is intentionally thin: it parses arguments, calls into
``research_navigator.ingest.pipeline``/``qdrant_store``, and renders output.
All business logic lives in ``src/research_navigator/ingest``.
"""

from __future__ import annotations

import argparse
import json
import sys

from research_navigator.config import get_settings
from research_navigator.ingest.pipeline import (
    IngestReport,
    ingest_corpus,
    reindex_corpus,
    validate_corpus,
)
from research_navigator.ingest.qdrant_store import QdrantStore
from research_navigator.logging import configure_logging


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _report_dict(report: IngestReport) -> dict[str, object]:
    return {
        "documents": [d.model_dump() for d in report.documents],
        "total_added": report.total_added,
        "total_deleted": report.total_deleted,
        "total_writes": report.total_writes,
        "errors": [d.doc_id for d in report.errors],
    }


def main(argv: list[str] | None = None) -> int:
    """Entrypoint. Returns a process exit code (non-zero on any document error)."""
    parser = argparse.ArgumentParser(prog="research_navigator.ingest")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Parse, chunk, embed, and upsert the corpus.")
    p_ingest.add_argument("--doc", help="Ingest only this doc_id.", default=None)
    p_ingest.add_argument(
        "--force", action="store_true", help="Re-embed and upsert even if unchanged."
    )
    sub.add_parser("validate", help="Parse + chunk every doc without writing; flag problems.")
    sub.add_parser("reindex", help="Drop the collection and re-ingest from scratch.")
    sub.add_parser("stats", help="Report chunk counts by content_type, year, and tags.")

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    if args.command == "ingest":
        report = ingest_corpus(settings, only_doc=args.doc, force=args.force)
        _print(_report_dict(report))
        return 1 if report.errors else 0
    if args.command == "validate":
        report = validate_corpus(settings)
        _print(_report_dict(report))
        return 1 if report.errors else 0
    if args.command == "reindex":
        report = reindex_corpus(settings)
        _print(_report_dict(report))
        return 1 if report.errors else 0
    if args.command == "stats":
        _print(QdrantStore(settings).stats())
        return 0
    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2


if __name__ == "__main__":
    sys.exit(main())
