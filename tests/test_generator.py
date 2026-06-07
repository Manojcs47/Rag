"""Unit tests for the generation backends (M2).

The extractive backend is exercised directly (offline, deterministic). The
OpenAI-compatible backend is tested via a stubbed ``httpx.post`` so no network is
touched and the marker-parsing / error paths are covered.
"""

from __future__ import annotations

import pytest

from research_navigator.config import GenerationConfig
from research_navigator.generate.citations import Citation
from research_navigator.generate.generator import (
    ExtractiveGenerator,
    GenerationError,
    OpenAICompatibleGenerator,
    build_generator,
)
from research_navigator.retrieve.hybrid import RetrievedChunk


def _source(number: int, text: str, doc_id: str = "d1") -> tuple[Citation, RetrievedChunk]:
    citation = Citation(
        number=number,
        doc_id=doc_id,
        title="T",
        authors_display="A et al.",
        year=2024,
        source="arXiv:0000.0000",
        section="Body",
        url="https://arxiv.org/abs/0000.0000",
    )
    chunk = RetrievedChunk(
        doc_id=doc_id, content_type="arxiv_paper", title="T", year=2024, text=text, score=1.0
    )
    return citation, chunk


def test_extractive_attaches_marker_to_every_sentence() -> None:
    gen = ExtractiveGenerator(max_sentences=4)
    sources = [
        _source(1, "Retrieval augmented generation grounds answers in documents.", "d1"),
        _source(2, "Generation improves when retrieval supplies relevant documents.", "d2"),
    ]
    out = gen.generate("what is retrieval augmented generation", sources)
    assert out.used_markers == [1, 2]
    assert "[1]" in out.text and "[2]" in out.text
    # Every sentence ends with a marker -> no unattributed claim.
    for sentence in out.text.split("] "):
        assert "[" in sentence or sentence.endswith("]")


def test_extractive_skips_sources_with_no_query_overlap() -> None:
    gen = ExtractiveGenerator(max_sentences=4)
    sources = [
        _source(1, "Retrieval augmented generation grounds answers in documents.", "d1"),
        _source(2, "Agents call tools to act in an environment.", "d2"),
    ]
    out = gen.generate("what is retrieval augmented generation", sources)
    # Source 2 shares no content words with the query, so it is not cited rather
    # than emitting an unrelated sentence (the cause of the garbled demo output).
    assert out.used_markers == [1]
    assert "[2]" not in out.text


def test_extractive_normalizes_whitespace_and_skips_fragments() -> None:
    gen = ExtractiveGenerator(max_sentences=2)
    # Layout noise: newlines mid-sentence plus a heading fragment that must not win.
    text = "Evaluation\nRetrieval augmented\ngeneration grounds answers in source documents."
    out = gen.generate("retrieval augmented generation", [_source(1, text)])
    assert "[1]" in out.text
    assert "\n" not in out.text  # newlines collapsed
    assert "Evaluation [1]" not in out.text  # bare heading fragment rejected


def test_extractive_picks_most_relevant_sentence() -> None:
    gen = ExtractiveGenerator(max_sentences=1)
    text = "Unrelated preamble about weather. The transformer uses self-attention over tokens."
    out = gen.generate("how does the transformer attention work", [_source(1, text)])
    assert "self-attention" in out.text
    assert "weather" not in out.text


def test_extractive_is_deterministic() -> None:
    gen = ExtractiveGenerator(max_sentences=3)
    sources = [_source(1, "Alpha sentence about scaling laws and models.")]
    assert gen.generate("scaling laws", sources) == gen.generate("scaling laws", sources)


def test_build_generator_defaults_to_extractive() -> None:
    gen = build_generator(_settings_with(GenerationConfig(backend="extractive")))
    assert gen.name == "extractive"


def test_openai_backend_parses_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "RAG grounds answers [1][2]."}}]}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())
    gen = OpenAICompatibleGenerator(GenerationConfig(backend="openai"))
    out = gen.generate("q", [_source(1, "x"), _source(2, "y", "d2")])
    assert out.used_markers == [1, 2]
    assert "grounds answers" in out.text


def test_openai_backend_raises_generation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    def _boom(*_a: object, **_k: object) -> object:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", _boom)
    gen = OpenAICompatibleGenerator(GenerationConfig(backend="openai"))
    with pytest.raises(GenerationError):
        gen.generate("q", [_source(1, "x")])


def _settings_with(generation: GenerationConfig):  # type: ignore[no-untyped-def]
    from research_navigator.config import Settings

    return Settings(generation=generation, use_offline_embedder=True)
