"""Integration tests for the ingest pipeline — the B4 acceptance criteria.

Covers: full corpus ingest, zero-write re-ingest (idempotency), modifying one
document updating only its chunks, validate, reindex, and error reporting.
All against in-memory Qdrant with the offline embedder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_navigator.config import Settings
from research_navigator.ingest.pipeline import (
    ingest_corpus,
    reindex_corpus,
    validate_corpus,
)
from research_navigator.ingest.qdrant_store import QdrantStore

pytestmark = pytest.mark.integration


def test_full_ingest_writes_chunks(settings: Settings, store: QdrantStore) -> None:
    report = ingest_corpus(settings, store=store)
    assert report.total_added > 0
    assert not report.errors
    assert store.stats()["total_chunks"] == report.total_added


def test_reingest_unchanged_is_zero_writes(settings: Settings, store: QdrantStore) -> None:
    first = ingest_corpus(settings, store=store)
    assert first.total_added > 0
    second = ingest_corpus(settings, store=store)
    # The headline B4 acceptance criterion: re-ingesting an unchanged corpus writes nothing.
    assert second.total_writes == 0
    assert all(d.skipped for d in second.documents)


def test_modifying_one_doc_updates_only_its_chunks(
    settings: Settings, store: QdrantStore, corpus_dir: Path
) -> None:
    ingest_corpus(settings, store=store)
    untouched_before = store.existing_point_ids("lillog-prompt-eng-2023-03")

    # Modify only the HF document on disk.
    hf_path = corpus_dir / "documents" / "hf-learn" / "hf-nlp-ch09.md"
    hf_path.write_text(
        hf_path.read_text(encoding="utf-8") + "\n\n## New Section\n\nFresh content added here.\n",
        encoding="utf-8",
    )

    report = ingest_corpus(settings, store=store)
    changed = {d.doc_id for d in report.documents if not d.skipped}
    assert changed == {"hf-nlp-ch09"}
    # The other document's points are byte-for-byte the same ids -> untouched.
    assert store.existing_point_ids("lillog-prompt-eng-2023-03") == untouched_before


def test_ingest_single_doc(settings: Settings, store: QdrantStore) -> None:
    report = ingest_corpus(settings, only_doc="hf-nlp-ch09", store=store)
    assert len(report.documents) == 1
    assert report.documents[0].doc_id == "hf-nlp-ch09"


def test_ingest_unknown_doc_raises(settings: Settings, store: QdrantStore) -> None:
    with pytest.raises(ValueError, match="doc_id not found"):
        ingest_corpus(settings, only_doc="nope", store=store)


def test_force_reingests_everything(settings: Settings, store: QdrantStore) -> None:
    ingest_corpus(settings, store=store)
    forced = ingest_corpus(settings, force=True, store=store)
    assert forced.total_added > 0


def test_validate_reports_chunk_counts_without_writing(settings: Settings) -> None:
    report = validate_corpus(settings)
    assert len(report.documents) == 2
    assert all(d.chunks > 0 for d in report.documents)
    assert not report.errors
    # validate must not create a collection / write anything.
    assert QdrantStore(settings).stats()["total_chunks"] == 0


def test_validate_flags_missing_file(settings: Settings, corpus_dir: Path) -> None:
    (corpus_dir / "documents" / "hf-learn" / "hf-nlp-ch09.md").unlink()
    report = validate_corpus(settings)
    errored = {d.doc_id for d in report.errors}
    assert "hf-nlp-ch09" in errored


def test_reindex_rebuilds_from_scratch(settings: Settings) -> None:
    report = reindex_corpus(settings)
    assert report.total_added > 0
    assert not report.errors


def test_ingest_continues_past_a_bad_document(
    settings: Settings, store: QdrantStore, corpus_dir: Path
) -> None:
    # Corrupt one file so its parse path is exercised but the run still completes.
    bad = corpus_dir / "documents" / "lillog" / "lillog-prompt-eng-2023-03.md"
    bad.unlink()
    bad.mkdir()  # a directory where a file is expected -> open() fails, reported not raised
    report = ingest_corpus(settings, store=store)
    errored = {d.doc_id for d in report.errors}
    assert "lillog-prompt-eng-2023-03" in errored
    # The healthy document still ingested.
    assert any(d.doc_id == "hf-nlp-ch09" and d.added > 0 for d in report.documents)
