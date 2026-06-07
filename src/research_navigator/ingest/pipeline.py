"""Ingestion orchestration: manifest -> parse -> chunk -> embed -> Qdrant.

Idempotency (B4): a chunk's point id is derived from ``(doc_id, chunk_index,
content_hash)``. Re-ingesting an unchanged corpus therefore finds every id already
present and writes nothing. Changing one document changes only that document's
hashes, so only its chunks are re-embedded and upserted, and any now-stale chunk
ids are deleted. No component fails silently — parse/embed errors per document are
logged and surfaced in the report; the run continues.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from research_navigator.config import Settings, get_settings
from research_navigator.ingest.chunk import Chunk, chunk_document
from research_navigator.ingest.embed import DenseEmbedder, SparseEmbedder, build_embedders
from research_navigator.ingest.manifest import DocumentMeta, Manifest, load_manifest
from research_navigator.ingest.parse import parse_document
from research_navigator.ingest.qdrant_store import QdrantStore
from research_navigator.logging import get_logger

log = get_logger(__name__)


class DocReport(BaseModel):
    """Per-document outcome of an ingest run."""

    doc_id: str
    chunks: int = 0
    added: int = 0
    deleted: int = 0
    skipped: bool = False
    error: str | None = None


class IngestReport(BaseModel):
    """Aggregate outcome of an ingest run."""

    documents: list[DocReport] = Field(default_factory=list)

    @property
    def total_added(self) -> int:
        return sum(d.added for d in self.documents)

    @property
    def total_deleted(self) -> int:
        return sum(d.deleted for d in self.documents)

    @property
    def total_writes(self) -> int:
        return self.total_added + self.total_deleted

    @property
    def errors(self) -> list[DocReport]:
        return [d for d in self.documents if d.error]


def _chunks_for(doc: DocumentMeta, settings: Settings) -> list[Chunk]:
    path = str(doc.resolved_path(settings.corpus_dir))
    parsed = parse_document(doc.doc_id, doc.content_type, path)
    params = settings.chunk.for_type(doc.content_type)
    return chunk_document(parsed, params)


def _ingest_one(
    doc: DocumentMeta,
    settings: Settings,
    store: QdrantStore,
    dense: DenseEmbedder,
    sparse: SparseEmbedder,
    force: bool,
) -> DocReport:
    try:
        chunks = _chunks_for(doc, settings)
    except Exception as exc:
        log.error("ingest_doc_failed", doc_id=doc.doc_id, error=str(exc))
        return DocReport(doc_id=doc.doc_id, error=str(exc))

    desired = {c.point_id: c for c in chunks}
    existing = set() if force else store.existing_point_ids(doc.doc_id)
    to_add = [c for c in chunks if force or c.point_id not in existing]
    to_delete = [pid for pid in existing if pid not in desired]

    if not to_add and not to_delete:
        log.info("ingest_doc_unchanged", doc_id=doc.doc_id, chunks=len(chunks))
        return DocReport(doc_id=doc.doc_id, chunks=len(chunks), skipped=True)

    added = 0
    if to_add:
        texts = [c.text for c in to_add]
        added = store.upsert_chunks(
            doc, to_add, dense.embed_passages(texts), sparse.embed_passages(texts)
        )
    deleted = store.delete_points(to_delete)
    log.info("ingest_doc_done", doc_id=doc.doc_id, chunks=len(chunks), added=added, deleted=deleted)
    return DocReport(doc_id=doc.doc_id, chunks=len(chunks), added=added, deleted=deleted)


def ingest_corpus(
    settings: Settings | None = None,
    *,
    only_doc: str | None = None,
    force: bool = False,
    store: QdrantStore | None = None,
) -> IngestReport:
    """Ingest the whole corpus (or a single doc) into Qdrant. Returns a report."""
    settings = settings or get_settings()
    manifest = load_manifest(settings.manifest_path)
    store = store or QdrantStore(settings)
    store.ensure_collection()
    dense, sparse = build_embedders(settings)

    docs = manifest.documents
    if only_doc is not None:
        docs = [d for d in docs if d.doc_id == only_doc]
        if not docs:
            raise ValueError(f"doc_id not found in manifest: {only_doc}")

    report = IngestReport(
        documents=[_ingest_one(d, settings, store, dense, sparse, force) for d in docs]
    )
    log.info(
        "ingest_complete",
        documents=len(report.documents),
        added=report.total_added,
        deleted=report.total_deleted,
        errors=len(report.errors),
    )
    return report


def validate_corpus(settings: Settings | None = None) -> IngestReport:
    """Parse + chunk every document without writing to Qdrant; flag problems."""
    settings = settings or get_settings()
    manifest = load_manifest(settings.manifest_path)
    reports: list[DocReport] = []
    for doc in manifest.documents:
        path = doc.resolved_path(settings.corpus_dir)
        if not Path(path).is_file():
            reports.append(DocReport(doc_id=doc.doc_id, error=f"missing file: {path}"))
            continue
        try:
            chunks = _chunks_for(doc, settings)
        except Exception as exc:
            reports.append(DocReport(doc_id=doc.doc_id, error=str(exc)))
            continue
        err = None if chunks else "produced 0 chunks"
        reports.append(DocReport(doc_id=doc.doc_id, chunks=len(chunks), error=err))
    return IngestReport(documents=reports)


def reindex_corpus(settings: Settings | None = None) -> IngestReport:
    """Drop the collection and re-ingest everything from scratch."""
    settings = settings or get_settings()
    store = QdrantStore(settings)
    store.drop_collection()
    return ingest_corpus(settings, force=True, store=store)


def corpus_manifest(settings: Settings | None = None) -> Manifest:
    """Convenience accessor for the validated manifest."""
    settings = settings or get_settings()
    return load_manifest(settings.manifest_path)
