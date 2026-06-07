"""Qdrant collection management and idempotent upserts (B3, B4, B5).

The collection holds one point per chunk with two named vectors (dense + sparse)
and a payload carrying *all* document-level manifest fields plus the per-chunk
fields (``section_title``, ``section_index``, ``chunk_index``, ``content_hash``).
Payload indexes are created on the high-cardinality filter fields so M2 can filter
inside Qdrant rather than in Python.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from qdrant_client import QdrantClient, models

from research_navigator.config import Settings
from research_navigator.ingest.chunk import Chunk
from research_navigator.ingest.embed import SparseVector
from research_navigator.ingest.manifest import DocumentMeta
from research_navigator.logging import get_logger

log = get_logger(__name__)

# Fields indexed for fast filtering inside Qdrant (B5).
_INDEXED_FIELDS: dict[str, models.PayloadSchemaType] = {
    "content_type": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
    "tags": models.PayloadSchemaType.KEYWORD,
    "primary_category": models.PayloadSchemaType.KEYWORD,
    "is_foundational": models.PayloadSchemaType.BOOL,
}


class QdrantStore:
    """Thin wrapper over :class:`QdrantClient` with our collection conventions."""

    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self._s = settings
        url = settings.qdrant_url
        if client is not None:
            self._c = client
        elif url == ":memory:":
            self._c = QdrantClient(location=":memory:")
        else:
            self._c = QdrantClient(url=url, timeout=settings.qdrant_timeout)

    @property
    def client(self) -> QdrantClient:
        return self._c

    # --- schema (B3, B5) ----------------------------------------------------
    def ensure_collection(self) -> None:
        """Create the collection + payload indexes if they do not already exist."""
        name = self._s.qdrant_collection
        if self._c.collection_exists(name):
            return
        self._c.create_collection(
            collection_name=name,
            vectors_config={
                self._s.dense_vector_name: models.VectorParams(
                    size=self._s.dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                self._s.sparse_vector_name: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Payload indexes have no effect.*")
            for field, schema in _INDEXED_FIELDS.items():
                self._c.create_payload_index(name, field_name=field, field_schema=schema)
        log.info("collection_created", collection=name, indexed_fields=list(_INDEXED_FIELDS))

    def drop_collection(self) -> None:
        """Delete the collection if present (used by ``reindex``)."""
        if self._c.collection_exists(self._s.qdrant_collection):
            self._c.delete_collection(self._s.qdrant_collection)
            log.info("collection_dropped", collection=self._s.qdrant_collection)

    # --- idempotency helpers (B4) ------------------------------------------
    def existing_point_ids(self, doc_id: str) -> set[str]:
        """Return the ids of all points currently stored for ``doc_id``."""
        ids: set[str] = set()
        offset: Any = None
        flt = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        while True:
            points, offset = self._c.scroll(
                collection_name=self._s.qdrant_collection,
                scroll_filter=flt,
                with_payload=False,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            ids.update(str(p.id) for p in points)
            if offset is None:
                break
        return ids

    def upsert_chunks(
        self,
        doc: DocumentMeta,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[SparseVector],
    ) -> int:
        """Upsert the given chunks for ``doc``. Returns the number of points written."""
        if not chunks:
            return 0
        points = [
            models.PointStruct(
                id=chunk.point_id,
                vector={
                    self._s.dense_vector_name: dvec,
                    self._s.sparse_vector_name: models.SparseVector(
                        indices=svec.indices, values=svec.values
                    ),
                },
                payload=_build_payload(doc, chunk),
            )
            for chunk, dvec, svec in zip(chunks, dense, sparse, strict=True)
        ]
        self._c.upsert(collection_name=self._s.qdrant_collection, points=points, wait=True)
        return len(points)

    def delete_points(self, ids: list[str]) -> int:
        """Delete points by id. Returns the count requested for deletion."""
        if not ids:
            return 0
        self._c.delete(
            collection_name=self._s.qdrant_collection,
            points_selector=models.PointIdsList(points=list(ids)),
            wait=True,
        )
        return len(ids)

    # --- stats (B6) ---------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Return chunk counts plus distributions by content_type, year, and tags."""
        name = self._s.qdrant_collection
        if not self._c.collection_exists(name):
            return {"total_chunks": 0, "by_content_type": {}, "by_year": {}, "by_tags": {}}
        by_ct: Counter[str] = Counter()
        by_year: Counter[int] = Counter()
        by_tags: Counter[str] = Counter()
        total = 0
        offset: Any = None
        while True:
            points, offset = self._c.scroll(
                collection_name=name,
                with_payload=["content_type", "year", "tags"],
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for p in points:
                total += 1
                payload = p.payload or {}
                by_ct[str(payload.get("content_type"))] += 1
                by_year[int(payload.get("year", 0))] += 1
                for tag in payload.get("tags", []) or []:
                    by_tags[str(tag)] += 1
            if offset is None:
                break
        return {
            "total_chunks": total,
            "by_content_type": dict(sorted(by_ct.items())),
            "by_year": dict(sorted(by_year.items())),
            "by_tags": dict(sorted(by_tags.items(), key=lambda kv: (-kv[1], kv[0]))),
        }


def _build_payload(doc: DocumentMeta, chunk: Chunk) -> dict[str, Any]:
    """All document-level manifest fields + per-chunk fields (B3)."""
    payload = doc.model_dump()
    payload.update(
        {
            "section_title": chunk.section_title,
            "section_index": chunk.section_index,
            "chunk_index": chunk.chunk_index,
            "content_hash": chunk.content_hash,
            "section_path": chunk.section_path,
            "kind": chunk.kind,
            "text": chunk.text,
        }
    )
    return payload
