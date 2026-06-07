"""Answer generation backends (M2, adr-0006).

Two interchangeable backends behind one :class:`Generator` protocol — mirroring the
embedder design (adr-0001):

  * :class:`ExtractiveGenerator` — offline, deterministic, dependency-free. It
    composes the answer from the most query-relevant sentence of each source and
    attaches that source's citation marker. It cannot fabricate (every sentence is
    drawn from a retrieved chunk and cited), which makes it the default for CI,
    tests, and a keyless demo.
  * :class:`OpenAICompatibleGenerator` — calls any OpenAI-compatible ``/chat/
    completions`` endpoint (local Ollama / vLLM / OpenAI) for fluent synthesis,
    constrained by the grounding system prompt. The pipeline still validates and
    deduplicates its citations, so a misbehaving model cannot smuggle in a fake one.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from research_navigator.config import GenerationConfig, Settings
from research_navigator.generate.citations import Citation
from research_navigator.generate.prompt import SYSTEM_PROMPT, build_user_prompt
from research_navigator.logging import get_logger
from research_navigator.retrieve.hybrid import RetrievedChunk

log = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z0-9]+")
_MARKER = re.compile(r"\[(\d+)\]")
# Minimal stopword set so overlap scoring tracks content words, not glue.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "what",
        "how",
        "why",
        "which",
        "who",
        "with",
        "as",
        "by",
        "at",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "about",
        "into",
        "from",
        "between",
        "compare",
        "explain",
        "describe",
        "tell",
        "me",
        "i",
    }
)


class GeneratedAnswer(BaseModel):
    """Raw output of a generator: answer text plus the markers it emitted."""

    text: str
    used_markers: list[int] = Field(default_factory=list)


@runtime_checkable
class Generator(Protocol):
    """Produces a citation-marked answer from a question and numbered sources."""

    name: str

    def generate(
        self, query: str, sources: list[tuple[Citation, RetrievedChunk]]
    ) -> GeneratedAnswer: ...


class GenerationError(RuntimeError):
    """Raised when a generation backend fails (network/LLM error). Never swallowed."""


def _content_terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}


def _sentences(text: str) -> list[str]:
    # Chunk text carries newlines and layout artifacts; collapse whitespace so the
    # splitter sees clean prose instead of one-token-per-line fragments.
    normalized = re.sub(r"\s+", " ", text).strip()
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(normalized) if s.strip()]
    return parts or ([normalized] if normalized else [])


def _is_wellformed(sentence: str) -> bool:
    """True for real prose; rejects headings/numbers/layout shards ('make.', 'Evaluation')."""
    words = _WORD.findall(sentence)
    return len(words) >= 5 and any(c.isalpha() for c in sentence)


# --------------------------------------------------------------------------- #
# Extractive (offline, deterministic)                                         #
# --------------------------------------------------------------------------- #
class ExtractiveGenerator:
    """Deterministic, non-fabricating generator: cite the best sentence per source."""

    name = "extractive"

    def __init__(self, max_sentences: int) -> None:
        self._max_sentences = max_sentences

    def _best_sentence(self, chunk_text: str, query_terms: set[str]) -> str:
        candidates = [s for s in _sentences(chunk_text) if _is_wellformed(s)]
        if not candidates:
            return ""
        scored = sorted(
            candidates,
            key=lambda s: (-len(_content_terms(s) & query_terms), candidates.index(s)),
        )
        best = scored[0]
        # No lexical overlap with the query means this chunk does not actually
        # address the question; skip it rather than emit an unrelated sentence.
        # (If every source is skipped the pipeline refuses, which is correct.)
        if not _content_terms(best) & query_terms:
            return ""
        return best

    def generate(
        self, query: str, sources: list[tuple[Citation, RetrievedChunk]]
    ) -> GeneratedAnswer:
        query_terms = _content_terms(query)
        pieces: list[str] = []
        used: list[int] = []
        for citation, chunk in sources[: self._max_sentences]:
            sentence = self._best_sentence(chunk.text, query_terms)
            if not sentence:
                continue
            sentence = sentence.rstrip()
            if sentence[-1:] not in ".!?":
                sentence += "."
            pieces.append(f"{sentence} [{citation.number}]")
            used.append(citation.number)
        text = " ".join(pieces)
        log.info("generate_extractive", sentences=len(pieces), markers=used)
        return GeneratedAnswer(text=text, used_markers=used)


# --------------------------------------------------------------------------- #
# OpenAI-compatible (Ollama / vLLM / OpenAI)                                  #
# --------------------------------------------------------------------------- #
class OpenAICompatibleGenerator:
    """Calls an OpenAI-compatible chat-completions endpoint (lazy ``httpx`` import)."""

    name = "openai"

    def __init__(self, cfg: GenerationConfig) -> None:
        self._cfg = cfg

    def generate(
        self, query: str, sources: list[tuple[Citation, RetrievedChunk]]
    ) -> GeneratedAnswer:
        import httpx  # lazy: only needed for this backend

        headers = {"Content-Type": "application/json"}
        if self._cfg.llm_api_key is not None:
            headers["Authorization"] = f"Bearer {self._cfg.llm_api_key.get_secret_value()}"
        payload: dict[str, Any] = {
            "model": self._cfg.llm_model,
            "temperature": self._cfg.llm_temperature,
            "max_tokens": self._cfg.llm_max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, sources)},
            ],
        }
        url = f"{self._cfg.llm_base_url.rstrip('/')}/chat/completions"
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=self._cfg.llm_timeout)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            text = str(data["choices"][0]["message"]["content"]).strip()
        except Exception as exc:
            log.error("generate_llm_failed", url=url, model=self._cfg.llm_model, error=str(exc))
            raise GenerationError(f"LLM generation failed: {exc}") from exc

        used = sorted({int(m.group(1)) for m in _MARKER.finditer(text)})
        log.info("generate_llm", model=self._cfg.llm_model, markers=used)
        return GeneratedAnswer(text=text, used_markers=used)


def build_generator(settings: Settings) -> Generator:
    """Construct the generation backend selected by ``settings``."""
    cfg = settings.generation
    if cfg.backend == "openai":
        log.info("generator_selected", backend="openai", model=cfg.llm_model)
        return OpenAICompatibleGenerator(cfg)
    log.info("generator_selected", backend="extractive")
    return ExtractiveGenerator(max_sentences=cfg.max_sentences)
