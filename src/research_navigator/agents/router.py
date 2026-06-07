"""Deterministic, heuristic query router for the LangGraph agent (M3).

The router classifies a free-text query into one of six routes and extracts the
hints downstream nodes need (paper doc_id, comparison targets, etc.). It is
intentionally **rule-based** rather than LLM-driven, for three reasons:

  1. Reproducibility (M5 engineering standards) — same input, same route.
  2. Zero secrets / zero network — same constraint as ADR-0001 / ADR-0006.
  3. Trivial unit-testability — every route covered by deterministic tests.

If a query is ambiguous, we route to ConceptExplanation rather than refuse; the
M2 refusal gate (ADR-0007) is the safety net for genuinely off-topic content.
"""

from __future__ import annotations

import re

from research_navigator.agents.state import AgentRoute, RoutingHints
from research_navigator.ingest.manifest import DocumentMeta

# --- intent signals ---------------------------------------------------------

_COMPARE_RE = re.compile(
    r"\b(compare|comparison|vs\.?|versus|differences?\s+between|"
    r"how\s+(?:does|do)\s+\w+\s+(?:compare|differ)|"
    r"what(?:'s|\s+is)\s+the\s+difference)\b",
    re.IGNORECASE,
)

_FIND_PAPERS_RE = re.compile(
    r"\b(recommend|reading\s+list|papers?\s+(?:on|about|for|to\s+read)|"
    r"where\s+(?:to|do\s+i)\s+start|"
    r"what\s+(?:are\s+the\s+)?(?:best|key|important|good)\s+papers?|"
    r"suggest\s+(?:some|me)?\s*papers?|"
    r"give\s+me\s+(?:some\s+)?papers?)\b",
    re.IGNORECASE,
)

_RECENT_RE = re.compile(
    r"\b(recent|recently|latest|newest|new\s+(?:work|developments?|papers?|results?)|"
    r"state[- ]of[- ]the[- ]art|cutting[- ]edge|"
    r"what(?:'s|\s+is)\s+new|this\s+year|last\s+year)\b",
    re.IGNORECASE,
)

_FOUNDATIONAL_RE = re.compile(
    r"\b(foundational|seminal|classic|landmark|canonical|original|"
    r"pioneering|where\s+(?:to|do\s+i)\s+start)\b",
    re.IGNORECASE,
)

_ARXIV_ID_RE = re.compile(r"\b(?:arxiv[: ]?)?(\d{4}\.\d{4,5})\b", re.IGNORECASE)

# Well-known methods/models that may not appear as canonical tags in the manifest.
# Keys are lowercase aliases; values are the display form.
KNOWN_METHODS: dict[str, str] = {
    "dpo": "DPO",
    "kto": "KTO",
    "simpo": "SimPO",
    "rlhf": "RLHF",
    "rlaif": "RLAIF",
    "crag": "CRAG",
    "raft": "RAFT",
    "graphrag": "GraphRAG",
    "graph rag": "GraphRAG",
    "lora": "LoRA",
    "react": "ReAct",
    "llama 2": "Llama 2",
    "llama 3": "Llama 3",
    "llama2": "Llama 2",
    "llama3": "Llama 3",
    "deepseek-v3": "DeepSeek-V3",
    "deepseek v3": "DeepSeek-V3",
    "deepseek-r1": "DeepSeek-R1",
    "deepseek r1": "DeepSeek-R1",
    "mixtral": "Mixtral",
    "qwen2": "Qwen2",
    "qwen 2": "Qwen2",
    "phi-3": "Phi-3",
    "phi 3": "Phi-3",
    "gemma 2": "Gemma 2",
    "gemini 1.5": "Gemini 1.5",
    "bert": "BERT",
    "gpt-3": "GPT-3",
    "instructgpt": "InstructGPT",
    "bitnet": "BitNet",
    "flashattention-3": "FlashAttention-3",
    "flashattention": "FlashAttention",
    "chain of thought": "Chain-of-Thought",
    "chain-of-thought": "Chain-of-Thought",
    "cot": "Chain-of-Thought",
    "constitutional ai": "Constitutional AI",
    "self-rewarding": "Self-Rewarding",
    "swe-agent": "SWE-Agent",
}

# Broad in-scope vocabulary (beyond canonical tags) so we don't misclassify
# AI/ML questions as out-of-scope just because no exact tag matched.
_IN_SCOPE_KEYWORDS = re.compile(
    r"\b(embed(?:ding)?s?|tokeniz(?:er|ation)|neural\s+network|deep\s+learning|"
    r"machine\s+learning|language\s+model|llms?|model|training|inference|"
    r"gradient|loss|attention|transformer|prompt(?:ing)?|fine.?tun(?:e|ing)|"
    r"agent|reasoning|alignment|safety|rag|retrieval|paper|research|"
    r"benchmark|evaluation|hallucinat(?:e|ion)|nlp|\bai\b|\bml\b|"
    r"reinforcement\s+learning|rl)\b",
    re.IGNORECASE,
)

_OFF_TOPIC = re.compile(
    r"\b(weather|recipe|cook|stock\s+price|sports?|football|cricket|"
    r"movie|song|president|election|capital\s+of|directions?\s+to|"
    r"flight|hotel|restaurant)\b",
    re.IGNORECASE,
)


def extract_comparison_targets(query: str, documents: list[DocumentMeta]) -> list[str]:
    """Return 2+ named methods/papers found in ``query`` (longest-alias-first)."""
    found: list[str] = []
    seen: set[str] = set()
    lower = query.lower()

    # Sort longest-first so "deepseek-v3" wins over "deepseek".
    methods_sorted = sorted(KNOWN_METHODS.items(), key=lambda kv: -len(kv[0]))
    for alias, canonical in methods_sorted:
        if re.search(rf"\b{re.escape(alias)}\b", lower) and canonical not in seen:
            seen.add(canonical)
            found.append(canonical)

    # If we still need more, scan title content words from the manifest.
    if len(found) < 2:
        for doc in documents:
            for word in re.findall(r"\b[A-Z][A-Za-z0-9-]{3,}\b", doc.title):
                if (
                    word.lower() in lower
                    and word not in seen
                    and word.lower() not in {"the", "and", "for", "with"}
                ):
                    seen.add(word)
                    found.append(word)
                    break
            if len(found) >= 4:
                break
    return found


def find_paper_reference(query: str, documents: list[DocumentMeta]) -> str | None:
    """Resolve a paper reference in ``query`` to a manifest doc_id, or ``None``."""
    lower = query.lower()

    # 1. Explicit arXiv id ("arxiv:2305.14314", "2305.14314")
    if m := _ARXIV_ID_RE.search(query):
        arxiv_id = m.group(1)
        for doc in documents:
            if arxiv_id in doc.doc_id:
                return doc.doc_id

    # 2. Distinctive capitalised title words (>=4 chars) from the manifest.
    for doc in documents:
        title_words = re.findall(r"\b[A-Z][A-Za-z0-9-]{3,}\b", doc.title)
        for word in title_words:
            if word.lower() in lower:
                return doc.doc_id

    # 3. Known method/model name -> first matching paper.
    methods_sorted = sorted(KNOWN_METHODS.items(), key=lambda kv: -len(kv[0]))
    for alias, canonical in methods_sorted:
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            for doc in documents:
                if canonical.lower() in doc.title.lower():
                    return doc.doc_id
    return None


def classify_query(query: str, documents: list[DocumentMeta]) -> tuple[AgentRoute, RoutingHints]:
    """Pick a route + extract hints for ``query``.

    The order below is the precedence. Earlier rules win when multiple match.
    """
    # 1. Comparison wins when 2+ identifiable targets are present.
    if _COMPARE_RE.search(query):
        targets = extract_comparison_targets(query, documents)
        if len(targets) >= 2:
            return AgentRoute.COMPARE_APPROACHES, RoutingHints(compare_targets=targets[:3])

    # 2. Find-papers wins when reading-list intent is explicit.
    if _FIND_PAPERS_RE.search(query):
        foundational = bool(_FOUNDATIONAL_RE.search(query))
        recent = bool(_RECENT_RE.search(query)) and not foundational
        return AgentRoute.FIND_PAPERS, RoutingHints(
            foundational_only=foundational, recent_only=recent
        )

    # 3. Recent developments.
    if _RECENT_RE.search(query):
        return AgentRoute.RECENT_DEVELOPMENTS, RoutingHints(recent_only=True)

    # 4. Paper deep dive when a specific paper is referenced.
    if paper := find_paper_reference(query, documents):
        return AgentRoute.PAPER_DEEP_DIVE, RoutingHints(paper_doc_id=paper)

    # 5. Out of scope: explicit off-topic markers and no in-scope signal.
    if _OFF_TOPIC.search(query) and not _IN_SCOPE_KEYWORDS.search(query):
        return AgentRoute.OUT_OF_SCOPE, RoutingHints()

    # 6. Default: concept explanation. The refusal gate handles edge cases.
    return AgentRoute.CONCEPT_EXPLANATION, RoutingHints()
