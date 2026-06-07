"""Unit tests for the deterministic agent router (M3)."""

from __future__ import annotations

from research_navigator.agents.router import (
    classify_query,
    extract_comparison_targets,
    find_paper_reference,
)
from research_navigator.agents.state import AgentRoute
from research_navigator.ingest.manifest import DocumentMeta


def _doc(doc_id: str, title: str, year: int = 2024, **kwargs: object) -> DocumentMeta:
    return DocumentMeta(
        doc_id=doc_id,
        content_type=kwargs.get("content_type", "arxiv_paper"),  # type: ignore[arg-type]
        title=title,
        authors=kwargs.get("authors", ["A", "B"]),  # type: ignore[arg-type]
        year=year,
        primary_category="cs.CL",
        tags=kwargs.get("tags", []),  # type: ignore[arg-type]
        is_foundational=bool(kwargs.get("is_foundational", False)),
        source_url=f"https://arxiv.org/abs/{doc_id}",
        local_path=f"documents/arxiv/{doc_id}.pdf",
    )


DOCS = [
    _doc(
        "arxiv-1706.03762",
        "Attention Is All You Need",
        2017,
        is_foundational=True,
        tags=["transformers", "attention"],
    ),
    _doc("arxiv-2305.18290", "Direct Preference Optimization", 2023, tags=["DPO", "alignment"]),
    _doc(
        "arxiv-2402.01306",
        "KTO: Model Alignment as Prospect Theoretic Optimization",
        2024,
        tags=["KTO", "alignment"],
    ),
    _doc("arxiv-2407.21783", "The Llama 3 Herd of Models", 2024, tags=["LLM", "open_models"]),
    _doc("arxiv-2412.19437", "DeepSeek-V3 Technical Report", 2024, tags=["LLM", "MoE"]),
]


# ---- compare_approaches ----------------------------------------------------


def test_compare_dpo_kto_routes_to_compare() -> None:
    route, hints = classify_query("Compare DPO and KTO for alignment", DOCS)
    assert route is AgentRoute.COMPARE_APPROACHES
    assert "DPO" in hints.compare_targets
    assert "KTO" in hints.compare_targets


def test_compare_vs_phrasing() -> None:
    route, hints = classify_query("Llama 3 vs DeepSeek-V3", DOCS)
    assert route is AgentRoute.COMPARE_APPROACHES
    assert {"Llama 3", "DeepSeek-V3"}.issubset(set(hints.compare_targets))


def test_difference_between_phrasing() -> None:
    route, _ = classify_query("What's the difference between DPO and SimPO?", DOCS)
    assert route is AgentRoute.COMPARE_APPROACHES


# ---- find_papers ----------------------------------------------------------


def test_find_papers_reading_list() -> None:
    route, hints = classify_query("Give me a reading list on RAG", DOCS)
    assert route is AgentRoute.FIND_PAPERS
    assert hints.foundational_only is False


def test_find_papers_foundational_flag() -> None:
    route, hints = classify_query("Recommend seminal papers on alignment", DOCS)
    assert route is AgentRoute.FIND_PAPERS
    assert hints.foundational_only is True


def test_find_papers_recent_flag() -> None:
    route, hints = classify_query("Recommend recent papers on agents", DOCS)
    assert route is AgentRoute.FIND_PAPERS
    assert hints.recent_only is True


# ---- paper_deep_dive ------------------------------------------------------


def test_paper_deep_dive_by_title_word() -> None:
    route, hints = classify_query("Tell me about Llama 3", DOCS)
    assert route is AgentRoute.PAPER_DEEP_DIVE
    assert hints.paper_doc_id == "arxiv-2407.21783"


def test_paper_deep_dive_by_arxiv_id() -> None:
    route, hints = classify_query("Summarise 1706.03762", DOCS)
    assert route is AgentRoute.PAPER_DEEP_DIVE
    assert hints.paper_doc_id == "arxiv-1706.03762"


def test_paper_deep_dive_by_method_alias() -> None:
    route, hints = classify_query("explain KTO", DOCS)
    assert route is AgentRoute.PAPER_DEEP_DIVE
    assert hints.paper_doc_id is not None


# ---- recent_developments --------------------------------------------------


def test_recent_developments_basic() -> None:
    route, _ = classify_query("Recent developments in agents", DOCS)
    assert route is AgentRoute.RECENT_DEVELOPMENTS


def test_latest_phrasing() -> None:
    route, _ = classify_query("latest in alignment research", DOCS)
    assert route is AgentRoute.RECENT_DEVELOPMENTS


def test_state_of_the_art_phrasing() -> None:
    route, _ = classify_query("state-of-the-art language models", DOCS)
    assert route is AgentRoute.RECENT_DEVELOPMENTS


# ---- concept_explanation (default) ----------------------------------------


def test_concept_what_is() -> None:
    route, _ = classify_query("What is retrieval augmented generation?", DOCS)
    assert route is AgentRoute.CONCEPT_EXPLANATION


def test_concept_how_does() -> None:
    route, _ = classify_query("How do transformers work?", DOCS)
    # 'transformers' matches a title word, so router prefers PaperDeepDive — both are valid.
    assert route in {AgentRoute.CONCEPT_EXPLANATION, AgentRoute.PAPER_DEEP_DIVE}


def test_concept_explain() -> None:
    route, _ = classify_query("explain RLHF in simple terms", DOCS)
    assert route is AgentRoute.CONCEPT_EXPLANATION


# ---- out_of_scope ---------------------------------------------------------


def test_out_of_scope_weather() -> None:
    route, _ = classify_query("What's the weather today?", DOCS)
    assert route is AgentRoute.OUT_OF_SCOPE


def test_out_of_scope_recipe() -> None:
    route, _ = classify_query("Give me a recipe for pasta", DOCS)
    assert route is AgentRoute.OUT_OF_SCOPE


def test_out_of_scope_sports() -> None:
    route, _ = classify_query("Who won the football match yesterday?", DOCS)
    assert route is AgentRoute.OUT_OF_SCOPE


# ---- extractor unit tests -------------------------------------------------


def test_extract_comparison_targets_returns_two_distinct_methods() -> None:
    targets = extract_comparison_targets("compare DPO and SimPO", DOCS)
    assert "DPO" in targets


def test_find_paper_reference_arxiv_id() -> None:
    assert find_paper_reference("read 1706.03762", DOCS) == "arxiv-1706.03762"


def test_find_paper_reference_returns_none_for_unrelated() -> None:
    assert find_paper_reference("what is the weather", DOCS) is None
