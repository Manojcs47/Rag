"""Central application configuration.

All tunable values (paths, service URLs, thresholds, model names) live here and
are read from the environment via ``pydantic-settings``. Nothing in ``src/`` should
hardcode a path, model, or threshold — import :func:`get_settings` instead.

Environment variables use the ``RN_`` prefix, e.g. ``RN_QDRANT_URL``. A local
``.env`` file is honoured (see ``.env.example``). Nested chunking parameters use
the ``__`` delimiter, e.g. ``RN_CHUNK__ARXIV_PAPER__MAX_TOKENS=600``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkParams(BaseModel):
    """Token budget for one content type's chunks (see chunking ADR)."""

    max_tokens: int = Field(default=512, ge=64, description="Hard upper bound per chunk.")
    overlap_tokens: int = Field(
        default=64, ge=0, description="Token overlap between adjacent chunks."
    )
    min_tokens: int = Field(
        default=32,
        ge=0,
        description="Chunks smaller than this are merged forward (avoids tiny fragments).",
    )


class ChunkingConfig(BaseModel):
    """Per-content-type chunking parameters. Defaults are defended in adr-0003."""

    arxiv_paper: ChunkParams = Field(default=ChunkParams(max_tokens=512, overlap_tokens=64))
    course_chapter: ChunkParams = Field(default=ChunkParams(max_tokens=384, overlap_tokens=48))
    survey_blog: ChunkParams = Field(default=ChunkParams(max_tokens=512, overlap_tokens=64))
    lab_blog_post: ChunkParams = Field(default=ChunkParams(max_tokens=512, overlap_tokens=64))

    def for_type(self, content_type: str) -> ChunkParams:
        """Return the params for ``content_type``, falling back to arXiv defaults."""
        return getattr(self, content_type, self.arxiv_paper)


class RetrievalConfig(BaseModel):
    """Hybrid-retrieval and refusal parameters (consumed by M2, see adr-0007)."""

    top_k: int = Field(
        default=5,
        ge=1,
        description="Number of deduplicated source documents surfaced to generation.",
    )
    candidate_k: int = Field(
        default=20,
        ge=1,
        description="Chunks fetched (post-fusion) before per-document deduplication.",
    )
    prefetch_limit: int = Field(
        default=40,
        ge=1,
        description="Candidates each retrieval branch (dense, sparse) returns before RRF fusion.",
    )
    refusal_min_score: float = Field(
        default=0.6,
        description=(
            "Minimum top dense cosine similarity required to attempt an answer. "
            "Below this the pipeline refuses. Tuned in adr-0007; the fused RRF score "
            "is rank-based and unsuitable as a confidence gate, so the dense cosine "
            "is used instead. Calibrated for bge-small-en-v1.5 *with* its query "
            "instruction prefix (see RN_QUERY_PREFIX); re-tune if you swap the model."
        ),
    )
    recent_year_floor: int = Field(
        default=2024,
        description="Year that an unqualified 'recent'/'latest' query maps to as a lower bound.",
    )


class GenerationConfig(BaseModel):
    """Answer-generation parameters (consumed by M2, see adr-0006)."""

    backend: Literal["extractive", "openai"] = Field(
        default="extractive",
        description=(
            "Generation backend. 'extractive' is offline, deterministic, and never "
            "fabricates (default for CI/tests/demo without keys). 'openai' calls any "
            "OpenAI-compatible chat endpoint (Ollama, vLLM, OpenAI) for synthesis."
        ),
    )
    max_sentences: int = Field(
        default=6,
        ge=1,
        description="Upper bound on sentences in an extractive answer.",
    )
    # --- OpenAI-compatible backend (only used when backend == 'openai') -------
    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL of the OpenAI-compatible chat API (default: local Ollama).",
    )
    llm_model: str = Field(
        default="llama3.1:8b-instruct-q4_K_M",
        description="Model id passed to the chat endpoint.",
    )
    llm_api_key: SecretStr | None = Field(
        default=None,
        description="Bearer token for the chat endpoint (omit for keyless local servers).",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Sampling temperature. 0.0 keeps answers reproducible (M5).",
    )
    llm_max_tokens: int = Field(default=800, ge=1, description="Max tokens to generate per answer.")
    llm_timeout: float = Field(
        default=60.0, gt=0.0, description="HTTP timeout (seconds) for a generation call."
    )


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_prefix="RN_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # --- Corpus (kept OUT of git; path is configurable so the corpus is a
    #     variable, not a constant — see the assignment brief). ----------------
    corpus_dir: Path = Field(
        default=Path("ai-research-navigator-corpus"),
        description="Root of the unpacked corpus package.",
    )
    manifest_path: Path = Field(
        default=Path("ai-research-navigator-corpus/manifest.json"),
        description="Path to the corpus manifest.json.",
    )

    # --- Qdrant -------------------------------------------------------------
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Base URL of the Qdrant REST API. Use ':memory:' for an in-process store.",
    )
    qdrant_collection: str = Field(
        default="research_navigator",
        description="Name of the Qdrant collection holding all chunks.",
    )
    qdrant_timeout: int = Field(default=60, description="Qdrant client timeout (seconds).")

    # --- Embeddings (ADR-0001: local open-source via sentence-transformers) --
    dense_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="sentence-transformers model id for dense passage/query embeddings.",
    )
    dense_dim: int = Field(default=384, ge=1, description="Dense embedding dimensionality.")
    dense_device: str = Field(default="cpu", description="Torch device for the dense model.")
    embed_batch_size: int = Field(default=64, ge=1, description="Batch size when encoding chunks.")
    # bge passages need no prefix; e5 passages need 'passage: '.
    passage_prefix: str = Field(
        default="", description="String prepended to passages before encoding."
    )
    # bge retrieval REQUIRES a query instruction prefix; without it query/passage
    # vectors are poorly separated and off-topic queries score deceptively high
    # (breaking the refusal gate). e5 uses 'query: '. Empty for prefix-free models.
    query_prefix: str = Field(
        default="Represent this sentence for searching relevant passages: ",
        description="String prepended to a query before encoding (model-specific).",
    )
    use_offline_embedder: bool = Field(
        default=False,
        description="Use the deterministic hashing embedder (no model download). For CI/tests.",
    )

    # --- Sparse retrieval (ADR-0004: TF vectors + Qdrant IDF == BM25) --------
    sparse_vector_name: str = Field(default="sparse", description="Named sparse vector in Qdrant.")
    dense_vector_name: str = Field(default="dense", description="Named dense vector in Qdrant.")
    sparse_vocab_size: int = Field(
        default=2**18,
        ge=1024,
        description="Hashed sparse vocabulary size (bucket count for term hashing).",
    )

    # --- Chunking (ADR-0003) ------------------------------------------------
    chunk: ChunkingConfig = Field(default=ChunkingConfig())

    # --- Retrieval + refusal (M2, ADR-0007) ---------------------------------
    retrieval: RetrievalConfig = Field(default=RetrievalConfig())

    # --- Generation (M2, ADR-0006) ------------------------------------------
    generation: GenerationConfig = Field(default=GenerationConfig())

    # --- Reproducibility ----------------------------------------------------
    seed: int = Field(default=42, description="Global seed for any stochastic step.")

    # --- Logging ------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Root log level.")
    json_logs: bool = Field(
        default=False,
        description="Emit JSON logs (True) or human-readable console logs (False).",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached, validated :class:`Settings` instance."""
    return Settings()
