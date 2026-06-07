"""Unit tests for embedders (offline/hashing path — no model downloads)."""

from __future__ import annotations

import math

from research_navigator.config import Settings
from research_navigator.ingest.embed import (
    DenseEmbedder,
    HashingDenseEmbedder,
    HashingSparseEmbedder,
    SparseEmbedder,
    SparseVector,
    build_embedders,
)


def test_dense_embedder_dimension_and_count() -> None:
    emb = HashingDenseEmbedder(dim=64)
    vecs = emb.embed_passages(["hello world", "another text"])
    assert len(vecs) == 2
    assert all(len(v) == 64 for v in vecs)


def test_dense_embedder_is_deterministic() -> None:
    emb = HashingDenseEmbedder(dim=64)
    assert emb.embed_query("speculative decoding") == emb.embed_query("speculative decoding")


def test_dense_vectors_are_l2_normalized() -> None:
    emb = HashingDenseEmbedder(dim=64)
    v = emb.embed_query("transformers attention mechanism")
    norm = math.sqrt(sum(x * x for x in v))
    assert math.isclose(norm, 1.0, rel_tol=1e-6)


def test_dense_empty_text_safe() -> None:
    emb = HashingDenseEmbedder(dim=32)
    v = emb.embed_query("")
    # Zero-token text must not divide by zero; returns a finite vector.
    assert len(v) == 32
    assert all(math.isfinite(x) for x in v)


def test_sparse_embedder_indices_sorted_and_unique() -> None:
    emb = HashingSparseEmbedder(vocab_size=1024)
    sv = emb.embed_query("alpha beta beta gamma")
    assert isinstance(sv, SparseVector)
    assert sv.indices == sorted(sv.indices)
    assert len(sv.indices) == len(set(sv.indices))
    assert len(sv.indices) == len(sv.values)


def test_sparse_term_frequency_is_log_scaled() -> None:
    emb = HashingSparseEmbedder(vocab_size=2**16)
    # A term repeated 3x -> value 1 + ln(3); a singleton -> value 1.0.
    sv = emb.embed_query("repeat repeat repeat once")
    assert max(sv.values) > 1.0
    assert min(sv.values) == 1.0


def test_sparse_empty_text_returns_empty_vector() -> None:
    emb = HashingSparseEmbedder(vocab_size=1024)
    sv = emb.embed_query("!!! ??? ...")  # no word tokens
    assert sv.indices == []
    assert sv.values == []


def test_build_embedders_offline_selects_hashing() -> None:
    settings = Settings(use_offline_embedder=True, dense_dim=48)
    dense, sparse = build_embedders(settings)
    assert isinstance(dense, HashingDenseEmbedder)
    assert isinstance(sparse, HashingSparseEmbedder)
    assert dense.dim == 48
    # Protocol conformance (runtime-checkable).
    assert isinstance(dense, DenseEmbedder)
    assert isinstance(sparse, SparseEmbedder)
