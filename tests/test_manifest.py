"""Unit tests for manifest loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from research_navigator.ingest.manifest import (
    CONTENT_TYPES,
    DocumentMeta,
    Manifest,
    load_manifest,
)


def test_load_manifest_parses_fixture(corpus_dir: Path) -> None:
    manifest = load_manifest(corpus_dir / "manifest.json")
    assert manifest.schema_version == "1.0"
    assert len(manifest.documents) == 2


def test_load_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "does-not-exist.json")


def test_load_manifest_bad_schema_raises(tmp_path: Path) -> None:
    bad = tmp_path / "manifest.json"
    # Missing required top-level "documents" key.
    bad.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_manifest(bad)


def test_by_id_indexes_documents(corpus_dir: Path) -> None:
    manifest = load_manifest(corpus_dir / "manifest.json")
    by_id = manifest.by_id()
    assert set(by_id) == {"hf-nlp-ch09", "lillog-prompt-eng-2023-03"}
    assert by_id["hf-nlp-ch09"].content_type == "course_chapter"


def test_resolved_path_joins_corpus_dir() -> None:
    doc = DocumentMeta(
        doc_id="d1",
        content_type="arxiv_paper",
        title="T",
        year=2024,
        primary_category="cs.CL",
        source_url="https://example.com",
        local_path="documents/arxiv/d1.pdf",
    )
    resolved = doc.resolved_path(Path("/corpus"))
    assert resolved == Path("/corpus/documents/arxiv/d1.pdf")


def test_nullable_fields_default_to_none() -> None:
    doc = DocumentMeta(
        doc_id="d1",
        content_type="course_chapter",
        title="T",
        year=2023,
        primary_category="course",
        source_url="https://example.com",
        local_path="documents/x.md",
    )
    # month and citation_count are nullable in the real corpus.
    assert doc.month is None
    assert doc.citation_count is None
    assert doc.is_foundational is False


def test_content_types_constant_matches_corpus() -> None:
    assert {
        "arxiv_paper",
        "course_chapter",
        "survey_blog",
        "lab_blog_post",
    } == CONTENT_TYPES


def test_manifest_model_round_trips() -> None:
    m = Manifest(
        schema_version="1.0",
        generated_for="tests",
        documents=[
            DocumentMeta(
                doc_id="d1",
                content_type="arxiv_paper",
                title="T",
                year=2024,
                primary_category="cs.CL",
                source_url="https://example.com",
                local_path="x.pdf",
            )
        ],
    )
    again = Manifest.model_validate(m.model_dump())
    assert again.documents[0].doc_id == "d1"
