"""Embedding: chunks -> dense vectors + sparse (BM25-style) vectors.

ADR-0001 (dense): local open-source models via ``sentence-transformers``
(default ``BAAI/bge-small-en-v1.5``). ADR-0004 (sparse): term-frequency vectors
whose IDF is computed by Qdrant's ``Modifier.IDF`` — i.e. BM25 without a separate
model download, fully deterministic.

Both kinds expose an offline, hashing-based fallback so tests / CI / air-gapped
environments never need to fetch a model. Select it with ``RN_USE_OFFLINE_EMBEDDER=1``.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from research_navigator.config import Settings
from research_navigator.logging import get_logger

log = get_logger(__name__)
_WORD = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class SparseVector:
    """A sparse vector as parallel index/value arrays (Qdrant's representation)."""

    indices: list[int]
    values: list[float]


@runtime_checkable
class DenseEmbedder(Protocol):
    """Produces a dense vector per text. Passages and queries may differ (prefixes)."""

    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class SparseEmbedder(Protocol):
    """Produces a sparse term-frequency vector per text."""

    def embed_passages(self, texts: list[str]) -> list[SparseVector]: ...
    def embed_query(self, text: str) -> SparseVector: ...


# --------------------------------------------------------------------------- #
# Dense                                                                       #
# --------------------------------------------------------------------------- #
class SentenceTransformerEmbedder:
    """Dense embedder backed by ``sentence-transformers`` (lazy import)."""

    def __init__(
        self,
        model_name: str,
        device: str,
        passage_prefix: str,
        query_prefix: str,
        batch_size: int,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._passage_prefix = passage_prefix
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self.dim = int(self._model.get_sentence_embedding_dimension())
        log.info("dense_model_loaded", model=model_name, dim=self.dim, device=device)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode([f"{self._passage_prefix}{t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"{self._query_prefix}{text}"])[0]


class HashingDenseEmbedder:
    """Deterministic offline dense embedder (feature hashing + L2 norm). No downloads."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _WORD.findall(text.lower()):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "big")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[h % self.dim] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


# --------------------------------------------------------------------------- #
# Sparse (BM25 via Qdrant IDF)                                                 #
# --------------------------------------------------------------------------- #
class HashingSparseEmbedder:
    """Term-frequency sparse vectors with hashed vocab; IDF is applied by Qdrant."""

    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _WORD.findall(text.lower())

    def _index(self, term: str) -> int:
        return int.from_bytes(hashlib.md5(term.encode()).digest()[:8], "big") % self._vocab_size

    def _vector(self, text: str) -> SparseVector:
        counts: Counter[int] = Counter(self._index(t) for t in self._tokenize(text))
        if not counts:
            return SparseVector(indices=[], values=[])
        items = sorted(counts.items())
        # log-scaled term frequency; Qdrant's IDF modifier supplies the IDF term.
        return SparseVector(
            indices=[i for i, _ in items],
            values=[1.0 + math.log(c) for _, c in items],
        )

    def embed_passages(self, texts: list[str]) -> list[SparseVector]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._vector(text)


def build_embedders(settings: Settings) -> tuple[DenseEmbedder, SparseEmbedder]:
    """Construct the dense + sparse embedders selected by ``settings``."""
    sparse = HashingSparseEmbedder(vocab_size=settings.sparse_vocab_size)
    if settings.use_offline_embedder:
        log.warning("using_offline_dense_embedder", dim=settings.dense_dim)
        return HashingDenseEmbedder(dim=settings.dense_dim), sparse
    dense = SentenceTransformerEmbedder(
        model_name=settings.dense_model,
        device=settings.dense_device,
        passage_prefix=settings.passage_prefix,
        query_prefix=settings.query_prefix,
        batch_size=settings.embed_batch_size,
    )
    return dense, sparse
