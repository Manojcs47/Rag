"""Integration tests for the Qdrant store (in-memory Qdrant; no server needed)."""

from __future__ import annotations

import pytest

from research_navigator.config import Settings
from research_navigator.ingest.chunk import Chunk
from research_navigator.ingest.embed import build_embedders
from research_navigator.ingest.manifest import DocumentMeta
from research_navigator.ingest.qdrant_store import QdrantStore

pytestmark = pytest.mark.integration


def _doc(doc_id: str = "d1") -> DocumentMeta:
    return DocumentMeta(
        doc_id=doc_id,
        content_type="arxiv_paper",
        title="A Paper",
        authors=["Ada", "Bob", "Cara"],
        year=2024,
        primary_category="cs.CL",
        tags=["LLM", "scaling"],
        is_foundational=True,
        source_url="https://arxiv.org/abs/0000.0000",
        local_path="documents/arxiv/d1.pdf",
    )


def _chunk(doc_id: str, idx: int, text: str) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        chunk_index=idx,
        section_title=f"S{idx}",
        section_index=idx,
        section_path=[f"S{idx}"],
        text=text,
        content_hash=f"hash{idx}",
    )


def _embed(store: QdrantStore, settings: Settings, chunks: list[Chunk]):
    dense, sparse = build_embedders(settings)
    texts = [c.text for c in chunks]
    return dense.embed_passages(texts), sparse.embed_passages(texts)


def test_ensure_collection_idempotent(store: QdrantStore) -> None:
    store.ensure_collection()
    store.ensure_collection()  # second call must not raise
    assert store.client.collection_exists(store._s.qdrant_collection)


def test_upsert_and_existing_ids(store: QdrantStore, settings: Settings) -> None:
    store.ensure_collection()
    doc = _doc()
    chunks = [_chunk("d1", 0, "alpha text"), _chunk("d1", 1, "beta text")]
    dvecs, svecs = _embed(store, settings, chunks)
    written = store.upsert_chunks(doc, chunks, dvecs, svecs)
    assert written == 2
    existing = store.existing_point_ids("d1")
    assert existing == {c.point_id for c in chunks}


def test_delete_points(store: QdrantStore, settings: Settings) -> None:
    store.ensure_collection()
    doc = _doc()
    chunks = [_chunk("d1", 0, "alpha"), _chunk("d1", 1, "beta")]
    dvecs, svecs = _embed(store, settings, chunks)
    store.upsert_chunks(doc, chunks, dvecs, svecs)
    deleted = store.delete_points([chunks[0].point_id])
    assert deleted == 1
    assert store.existing_point_ids("d1") == {chunks[1].point_id}


def test_stats_distributions(store: QdrantStore, settings: Settings) -> None:
    store.ensure_collection()
    doc = _doc()
    chunks = [_chunk("d1", 0, "alpha"), _chunk("d1", 1, "beta")]
    dvecs, svecs = _embed(store, settings, chunks)
    store.upsert_chunks(doc, chunks, dvecs, svecs)
    stats = store.stats()
    assert stats["total_chunks"] == 2
    assert stats["by_content_type"] == {"arxiv_paper": 2}
    assert stats["by_year"] == {2024: 2}
    assert stats["by_tags"]["LLM"] == 2
    assert stats["by_tags"]["scaling"] == 2


def test_payload_carries_all_manifest_fields(store: QdrantStore, settings: Settings) -> None:
    store.ensure_collection()
    doc = _doc()
    chunks = [_chunk("d1", 0, "alpha text payload check")]
    dvecs, svecs = _embed(store, settings, chunks)
    store.upsert_chunks(doc, chunks, dvecs, svecs)
    points, _ = store.client.scroll(
        collection_name=settings.qdrant_collection, with_payload=True, limit=1
    )
    payload = points[0].payload or {}
    # Document-level fields (B3)...
    for field in ("doc_id", "content_type", "title", "year", "tags", "is_foundational"):
        assert field in payload
    # ...plus per-chunk fields.
    for field in ("section_title", "section_index", "chunk_index", "content_hash", "kind", "text"):
        assert field in payload


def test_drop_collection(store: QdrantStore) -> None:
    store.ensure_collection()
    assert store.client.collection_exists(store._s.qdrant_collection)
    store.drop_collection()
    assert not store.client.collection_exists(store._s.qdrant_collection)


def test_stats_on_missing_collection_is_empty(settings: Settings) -> None:
    fresh = QdrantStore(settings)
    stats = fresh.stats()
    assert stats == {"total_chunks": 0, "by_content_type": {}, "by_year": {}, "by_tags": {}}
