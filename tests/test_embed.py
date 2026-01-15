"""Tests for embedding encoders and disk cache."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ads_core.embed.encoder import StubEncoder, Encoder
from ads_core.embed.cache import DiskEmbeddingCache, CacheStats
from ads_core.utils.hashing import sha256_text


class TestStubEncoder:
    """Tests for StubEncoder."""

    def test_stub_encoder_model_id(self):
        """Test stub encoder has correct model_id."""
        enc = StubEncoder(d=64)
        assert enc.model_id == "stub-encoder-v0"

    def test_stub_encoder_custom_model_id(self):
        """Test stub encoder with custom model_id."""
        enc = StubEncoder(d=32, model_id="stub-v1")
        assert enc.model_id == "stub-v1"

    def test_stub_encoder_output_shape(self):
        """Test stub encoder produces correct shape."""
        enc = StubEncoder(d=64)
        texts = ["Hello world", "Test sentence", "Another one"]
        vecs = enc.encode(texts)

        assert vecs.shape == (3, 64)
        assert vecs.dtype == np.float32

    def test_stub_encoder_normalized(self):
        """Test stub encoder produces normalized vectors."""
        enc = StubEncoder(d=64)
        vecs = enc.encode(["Test"])
        norm = np.linalg.norm(vecs[0])
        assert abs(norm - 1.0) < 1e-5

    def test_stub_encoder_deterministic(self):
        """Test stub encoder is deterministic."""
        enc = StubEncoder(d=64)
        vecs1 = enc.encode(["Hello world"])
        vecs2 = enc.encode(["Hello world"])
        np.testing.assert_array_equal(vecs1, vecs2)

    def test_stub_encoder_empty_input(self):
        """Test stub encoder handles empty input."""
        enc = StubEncoder(d=64)
        vecs = enc.encode([])
        assert vecs.shape == (0, 64)


class TestCacheStats:
    """Tests for CacheStats."""

    def test_cache_stats_hit_rate(self):
        """Test cache stats hit rate calculation."""
        stats = CacheStats(hits=7, misses=3, total=10)
        assert stats.hit_rate == 0.7

    def test_cache_stats_zero_total(self):
        """Test cache stats with zero total."""
        stats = CacheStats(hits=0, misses=0, total=0)
        assert stats.hit_rate == 0.0

    def test_cache_stats_to_dict(self):
        """Test cache stats serialization."""
        stats = CacheStats(hits=5, misses=5, total=10)
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["total"] == 10
        assert d["hit_rate"] == 0.5


class TestDiskEmbeddingCache:
    """Tests for DiskEmbeddingCache."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        return tmp_path / "embeddings"

    @pytest.fixture
    def encoder(self):
        """Create a stub encoder."""
        return StubEncoder(d=32, model_id="test-encoder-v1")

    def test_cache_initialization(self, cache_dir):
        """Test cache initialization."""
        cache = DiskEmbeddingCache(root=cache_dir, model_id="test-model")
        assert not cache.exists()
        assert cache.model_id == "test-model"

    def test_cache_safe_model_id(self, cache_dir):
        """Test model_id is sanitized for filesystem."""
        cache = DiskEmbeddingCache(root=cache_dir, model_id="sbert:my/model:v1")
        safe = cache._safe_model_id()
        assert "/" not in safe
        assert ":" not in safe

    def test_cache_miss_computes(self, cache_dir, encoder):
        """Test cache computes on miss."""
        cache = DiskEmbeddingCache(root=cache_dir, model_id=encoder.model_id)
        texts = ["Hello world", "Test sentence"]

        vecs = cache.get_or_compute(texts, encoder.encode)

        assert vecs.shape == (2, 32)
        assert cache.exists()
        assert cache.last_stats.misses == 2
        assert cache.last_stats.hits == 0

    def test_cache_hit_reuses(self, cache_dir, encoder):
        """Test cache reuses on hit."""
        cache = DiskEmbeddingCache(root=cache_dir, model_id=encoder.model_id)
        texts = ["Hello world", "Test sentence"]

        # First call - compute
        vecs1 = cache.get_or_compute(texts, encoder.encode)
        assert cache.last_stats.misses == 2

        # Second call - should hit cache
        vecs2 = cache.get_or_compute(texts, encoder.encode)
        assert cache.last_stats.hits == 2
        assert cache.last_stats.misses == 0

        np.testing.assert_array_equal(vecs1, vecs2)

    def test_cache_partial_hit(self, cache_dir, encoder):
        """Test cache with partial hits."""
        cache = DiskEmbeddingCache(root=cache_dir, model_id=encoder.model_id)

        # First call with 2 texts
        vecs1 = cache.get_or_compute(["Hello", "World"], encoder.encode)
        assert cache.last_stats.misses == 2

        # Second call with 1 new, 1 cached
        vecs2 = cache.get_or_compute(["Hello", "New text"], encoder.encode)
        assert cache.last_stats.hits == 1
        assert cache.last_stats.misses == 1

        # Verify "Hello" gives same embedding
        np.testing.assert_array_equal(vecs1[0], vecs2[0])

    def test_cache_preserves_order(self, cache_dir, encoder):
        """Test cache preserves text order in output."""
        cache = DiskEmbeddingCache(root=cache_dir, model_id=encoder.model_id)

        texts = ["A", "B", "C"]
        vecs1 = cache.get_or_compute(texts, encoder.encode)

        # Request in different order
        vecs2 = cache.get_or_compute(["C", "A", "B"], encoder.encode)

        # C should match
        np.testing.assert_array_equal(vecs1[2], vecs2[0])
        # A should match
        np.testing.assert_array_equal(vecs1[0], vecs2[1])
        # B should match
        np.testing.assert_array_equal(vecs1[1], vecs2[2])

    def test_cache_metadata(self, cache_dir, encoder):
        """Test cache writes metadata."""
        cache = DiskEmbeddingCache(root=cache_dir, model_id=encoder.model_id)
        cache.get_or_compute(["test"], encoder.encode)

        meta = cache.load_meta()
        assert meta["model_id"] == encoder.model_id
        assert "created_at" in meta
        assert "updated_at" in meta
        assert meta["num_vectors"] == 1
        assert meta["vector_dim"] == 32

    def test_cache_info(self, cache_dir, encoder):
        """Test cache info method."""
        cache = DiskEmbeddingCache(root=cache_dir, model_id=encoder.model_id)
        cache.get_or_compute(["a", "b", "c"], encoder.encode)

        info = cache.get_info()
        assert info["model_id"] == encoder.model_id
        assert info["exists"] is True
        assert info["num_entries"] == 3

    def test_cache_clear(self, cache_dir, encoder):
        """Test cache clearing."""
        cache = DiskEmbeddingCache(root=cache_dir, model_id=encoder.model_id)
        cache.get_or_compute(["test"], encoder.encode)
        assert cache.exists()

        cache.clear()
        assert not cache.exists()

    def test_cache_different_models_separate(self, cache_dir):
        """Test different models have separate caches."""
        enc1 = StubEncoder(d=32, model_id="model-v1")
        enc2 = StubEncoder(d=32, model_id="model-v2")

        cache1 = DiskEmbeddingCache(root=cache_dir, model_id=enc1.model_id)
        cache2 = DiskEmbeddingCache(root=cache_dir, model_id=enc2.model_id)

        vecs1 = cache1.get_or_compute(["test"], enc1.encode)
        vecs2 = cache2.get_or_compute(["test"], enc2.encode)

        # Both should miss (different caches)
        assert cache1.last_stats.misses == 1
        assert cache2.last_stats.misses == 1

        # Verify they have separate storage
        assert cache1._cache_dir() != cache2._cache_dir()

    def test_cache_sha256_keying(self, cache_dir, encoder):
        """Test cache uses sha256 for keying."""
        cache = DiskEmbeddingCache(root=cache_dir, model_id=encoder.model_id)
        cache.get_or_compute(["Hello world"], encoder.encode)

        index = cache.load_index()
        expected_key = sha256_text("Hello world")
        assert expected_key in index


class TestCacheWithEncoder:
    """Integration tests for cache with different encoders."""

    def test_stub_encoder_with_cache(self, tmp_path):
        """Test stub encoder integrates with cache."""
        enc = StubEncoder(d=64)
        cache = DiskEmbeddingCache(root=tmp_path, model_id=enc.model_id)

        texts = ["Test 1", "Test 2", "Test 3"]

        # First run
        vecs1 = cache.get_or_compute(texts, enc.encode)
        assert cache.last_stats.hit_rate == 0.0

        # Second run (same texts)
        vecs2 = cache.get_or_compute(texts, enc.encode)
        assert cache.last_stats.hit_rate == 1.0

        np.testing.assert_array_equal(vecs1, vecs2)

    def test_cache_verbose_output(self, tmp_path, capsys):
        """Test cache verbose output."""
        enc = StubEncoder(d=32)
        cache = DiskEmbeddingCache(root=tmp_path, model_id=enc.model_id)

        cache.get_or_compute(["test"], enc.encode, verbose=True)
        captured = capsys.readouterr()
        assert "[cache]" in captured.out
        assert "hits=0" in captured.out
        assert "misses=1" in captured.out


class TestEncoderInterface:
    """Tests for Encoder interface compliance."""

    def test_stub_encoder_is_encoder(self):
        """Test StubEncoder implements Encoder interface."""
        enc = StubEncoder()
        assert isinstance(enc, Encoder)
        assert hasattr(enc, "model_id")
        assert hasattr(enc, "encode")
