"""Load and validate the corpus ``manifest.json`` into typed models.

The manifest is the single source of document-level metadata. Every field here is
carried through to every chunk's Qdrant payload (B3), so the model is intentionally
strict: an unexpected schema is a loud failure, not a silent ``KeyError`` later.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

CONTENT_TYPES = frozenset({"arxiv_paper", "course_chapter", "survey_blog", "lab_blog_post"})


class DocumentMeta(BaseModel):
    """One document's manifest entry. Field names mirror the manifest exactly."""

    doc_id: str
    content_type: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int
    month: int | None = None
    primary_category: str
    secondary_categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    is_foundational: bool = False
    citation_count: int | None = None
    source_url: str
    local_path: str

    def resolved_path(self, corpus_dir: Path) -> Path:
        """Absolute path to the document file on disk."""
        return corpus_dir / self.local_path


class Manifest(BaseModel):
    """Top-level manifest document."""

    schema_version: str
    generated_for: str
    documents: list[DocumentMeta]

    def by_id(self) -> dict[str, DocumentMeta]:
        """Index documents by ``doc_id``."""
        return {d.doc_id: d for d in self.documents}


def load_manifest(manifest_path: Path) -> Manifest:
    """Read and validate the manifest at ``manifest_path``.

    Raises:
        FileNotFoundError: if the manifest does not exist.
        pydantic.ValidationError: if the schema does not match :class:`Manifest`.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return Manifest.model_validate(raw)
