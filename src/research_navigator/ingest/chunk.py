"""Chunking: a :class:`ParsedDocument` -> a list of retrievable :class:`Chunk`.

Rules (defended in adr-0003):
  * The paper **abstract** is always its own chunk.
  * Chunks never cross **section boundaries**; within a section we size-bound to a
    token budget with a small overlap so adjacent chunks share context.
  * **Code fences** are atomic — never split mid-block. A code block larger than the
    budget becomes its own chunk rather than being broken.
  * **References** are excluded from retrieval entirely (they live on the parsed doc
    for citation lookup, never chunked).
  * Token counting is a dependency-free approximation (word/punct tokens). It is
    deterministic, which is what idempotency (B4) and reproducibility (M5) need.
"""

from __future__ import annotations

import hashlib
import re
import uuid

from pydantic import BaseModel, Field

from research_navigator.config import ChunkParams
from research_navigator.ingest.parse import ParsedDocument, Section

# Fixed namespace so UUIDv5 point ids are stable across runs (reproducibility).
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")

_TOKEN = re.compile(r"\w+|[^\w\s]")
_CODE_FENCE = re.compile(r"(```.*?```|~~~.*?~~~)", re.DOTALL)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class Chunk(BaseModel):
    """One retrievable unit with the metadata needed for filtering and citation."""

    doc_id: str
    chunk_index: int
    section_title: str
    section_index: int
    section_path: list[str] = Field(default_factory=list)
    kind: str = "body"  # "abstract" | "body"
    text: str
    content_hash: str

    @property
    def point_id(self) -> str:
        """Deterministic, Qdrant-valid (UUIDv5) chunk id from the chunk's identity -> B4."""
        raw = f"{self.doc_id}:{self.chunk_index}:{self.content_hash}"
        return str(uuid.uuid5(_POINT_NAMESPACE, raw))


def count_tokens(text: str) -> int:
    """Approximate token count (deterministic, model-independent)."""
    return len(_TOKEN.findall(text))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _segment(text: str) -> list[tuple[str, bool]]:
    """Split section text into (block, is_code) parts; code fences kept atomic."""
    parts: list[tuple[str, bool]] = []
    last = 0
    for m in _CODE_FENCE.finditer(text):
        if m.start() > last:
            parts.extend(
                (p.strip(), False) for p in text[last : m.start()].split("\n\n") if p.strip()
            )
        parts.append((m.group(0), True))
        last = m.end()
    if last < len(text):
        parts.extend((p.strip(), False) for p in text[last:].split("\n\n") if p.strip())
    return parts


def _split_oversized_prose(block: str, max_tokens: int) -> list[str]:
    """Break a prose block that exceeds the budget into sentence-packed pieces.

    Falls back to a hard word-window split for blocks with no sentence boundaries
    (e.g. Markdown tables), so no prose chunk ever exceeds ``max_tokens``.
    """
    sentences = _SENTENCE.split(block)
    pieces: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for sent in sentences:
        st = count_tokens(sent)
        if cur and cur_tokens + st > max_tokens:
            pieces.append(" ".join(cur))
            cur, cur_tokens = [], 0
        cur.append(sent)
        cur_tokens += st
    if cur:
        pieces.append(" ".join(cur))

    final: list[str] = []
    for piece in pieces:
        if count_tokens(piece) <= max_tokens:
            final.append(piece)
            continue
        words = piece.split()
        window: list[str] = []
        window_tokens = 0
        for word in words:
            wt = count_tokens(word)
            if window and window_tokens + wt > max_tokens:
                final.append(" ".join(window))
                window, window_tokens = [], 0
            window.append(word)
            window_tokens += wt
        if window:
            final.append(" ".join(window))
    return final


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    """Return a trailing slice of ``text`` containing at most ``overlap_tokens`` tokens."""
    if overlap_tokens <= 0:
        return ""
    words = text.split()
    tail: list[str] = []
    total = 0
    for word in reversed(words):
        wt = count_tokens(word)
        if tail and total + wt > overlap_tokens:
            break
        tail.append(word)
        total += wt
    return " ".join(reversed(tail))


def _chunk_section_text(text: str, params: ChunkParams) -> list[str]:
    """Pack one section's text into size-bounded chunks with word overlap."""
    blocks = _segment(text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    # Reserve room so a chunk + its overlap tail stays within max_tokens.
    content_budget = max(params.max_tokens - params.overlap_tokens, params.max_tokens // 2)

    def flush() -> None:
        nonlocal cur, cur_tokens
        if cur:
            chunks.append("\n\n".join(cur).strip())
            cur, cur_tokens = [], 0

    for content, is_code in blocks:
        sub_blocks = (
            [content]
            if is_code or count_tokens(content) <= content_budget
            else _split_oversized_prose(content, content_budget)
        )
        for sub in sub_blocks:
            st = count_tokens(sub)
            if cur and cur_tokens + st > content_budget:
                tail = _overlap_tail(
                    chunks[-1] if chunks else "\n\n".join(cur), params.overlap_tokens
                )
                flush()
                if tail:
                    cur.append(tail)
                    cur_tokens += count_tokens(tail)
            cur.append(sub)
            cur_tokens += st
    flush()

    # Merge a tiny trailing chunk back into its predecessor.
    if len(chunks) >= 2 and count_tokens(chunks[-1]) < params.min_tokens:
        chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}"
        chunks.pop()
    return chunks


def chunk_document(parsed: ParsedDocument, params: ChunkParams) -> list[Chunk]:
    """Produce the ordered chunks for a parsed document."""
    out: list[Chunk] = []
    idx = 0

    if parsed.abstract:
        text = parsed.abstract.strip()
        out.append(
            Chunk(
                doc_id=parsed.doc_id,
                chunk_index=idx,
                section_title="Abstract",
                section_index=-1,
                section_path=["Abstract"],
                kind="abstract",
                text=text,
                content_hash=_hash_text(text),
            )
        )
        idx += 1

    for s_idx, section in enumerate(parsed.sections):
        for piece in _chunk_section_text(section.text, params):
            out.append(_build_chunk(parsed.doc_id, idx, s_idx, section, piece))
            idx += 1
    return out


def _build_chunk(doc_id: str, idx: int, s_idx: int, section: Section, text: str) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        chunk_index=idx,
        section_title=section.title,
        section_index=s_idx,
        section_path=section.path,
        kind="body",
        text=text,
        content_hash=_hash_text(text),
    )
