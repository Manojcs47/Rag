"""Unit tests for chunking: abstract handling, token budgets, code atomicity, ids."""

from __future__ import annotations

from research_navigator.config import ChunkParams
from research_navigator.ingest.chunk import (
    Chunk,
    chunk_document,
    count_tokens,
)
from research_navigator.ingest.parse import ParsedDocument, Section


def _doc(**kw: object) -> ParsedDocument:
    base: dict[str, object] = {"doc_id": "d1", "content_type": "arxiv_paper"}
    base.update(kw)
    return ParsedDocument(**base)  # type: ignore[arg-type]


def test_count_tokens_is_deterministic() -> None:
    assert count_tokens("hello world") == 2
    assert count_tokens("don't.") == count_tokens("don't.")
    assert count_tokens("") == 0


def test_abstract_becomes_its_own_chunk() -> None:
    parsed = _doc(abstract="This is the abstract of the paper.", sections=[])
    chunks = chunk_document(parsed, ChunkParams())
    assert len(chunks) == 1
    assert chunks[0].kind == "abstract"
    assert chunks[0].section_title == "Abstract"
    assert chunks[0].chunk_index == 0


def test_chunks_respect_token_budget() -> None:
    long_text = " ".join(f"word{i}" for i in range(2000))
    parsed = _doc(sections=[Section(title="S", level=1, path=["S"], text=long_text)])
    params = ChunkParams(max_tokens=128, overlap_tokens=16, min_tokens=8)
    chunks = chunk_document(parsed, params)
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= params.max_tokens


def test_code_block_kept_intact() -> None:
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(50)) + "\n```"
    parsed = _doc(sections=[Section(title="S", level=1, path=["S"], text=code)])
    chunks = chunk_document(parsed, ChunkParams(max_tokens=64))
    # The fenced block survives in exactly one chunk, not split across many.
    holding = [c for c in chunks if "```python" in c.text]
    assert len(holding) == 1
    assert "x49 = 49" in holding[0].text


def test_point_id_deterministic_and_uuid() -> None:
    c = Chunk(
        doc_id="d1",
        chunk_index=3,
        section_title="S",
        section_index=0,
        text="hello",
        content_hash="abc123",
    )
    # Stable across calls; valid UUID string (Qdrant requires UUID or uint).
    assert c.point_id == c.point_id
    assert len(c.point_id) == 36
    assert c.point_id.count("-") == 4


def test_point_id_changes_with_content_hash() -> None:
    a = Chunk(
        doc_id="d", chunk_index=0, section_title="s", section_index=0, text="x", content_hash="h1"
    )
    b = Chunk(
        doc_id="d", chunk_index=0, section_title="s", section_index=0, text="x", content_hash="h2"
    )
    assert a.point_id != b.point_id


def test_chunk_indices_are_sequential() -> None:
    parsed = _doc(
        abstract="An abstract long enough to be kept as a chunk on its own here.",
        sections=[
            Section(title="A", level=1, path=["A"], text="Alpha content here."),
            Section(title="B", level=1, path=["B"], text="Beta content here."),
        ],
    )
    chunks = chunk_document(parsed, ChunkParams())
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_section_metadata_propagates() -> None:
    parsed = _doc(
        sections=[
            Section(title="Methods", level=2, path=["Paper", "Methods"], text="We did things.")
        ]
    )
    chunks = chunk_document(parsed, ChunkParams())
    assert chunks[0].section_title == "Methods"
    assert chunks[0].section_index == 0
    assert chunks[0].section_path == ["Paper", "Methods"]


def test_no_chunks_for_empty_document() -> None:
    assert chunk_document(_doc(sections=[]), ChunkParams()) == []
