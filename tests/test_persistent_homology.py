"""Tests for persistent homology module."""
import numpy as np
import pytest

from ads_core.topology.persistent_homology import (
    compute_persistence,
    compare_topologies,
    reduce_embeddings,
    TopologicalSummary,
    PersistenceFeature,
    _persistence_entropy,
)


@pytest.fixture
def circle_points():
    """Points on a circle — should have β₁=1 (one loop)."""
    t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)])


@pytest.fixture
def two_clusters():
    """Two well-separated clusters — should have β₀=2 at appropriate scale."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(30, 2) + np.array([0, 0])
    c2 = rng.randn(30, 2) + np.array([10, 10])
    return np.vstack([c1, c2])


@pytest.fixture
def high_dim_embeddings():
    """Synthetic 384d embeddings mimicking sentence-transformers output."""
    rng = np.random.RandomState(42)
    return rng.randn(100, 384).astype(np.float32)


class TestPersistenceFeature:
    def test_persistence(self):
        f = PersistenceFeature(dimension=1, birth=0.5, death=2.0)
        assert f.persistence == 1.5

    def test_midlife(self):
        f = PersistenceFeature(dimension=0, birth=1.0, death=3.0)
        assert f.midlife == 2.0


class TestComputePersistence:
    def test_circle_has_loop(self, circle_points):
        summary = compute_persistence(circle_points, max_dim=1, reduce_dim=None)
        assert summary.n_points == 50
        assert summary.embedding_dim == 2
        # Circle should have at least one H₁ feature
        h1_feats = summary.features_by_dim(1)
        assert len(h1_feats) > 0
        # The most persistent H₁ feature should be significant
        top = summary.top_persistent(1, k=1)
        assert len(top) == 1
        assert top[0].persistence > 0

    def test_two_clusters(self, two_clusters):
        summary = compute_persistence(two_clusters, max_dim=1, reduce_dim=None)
        assert summary.n_points == 60
        # Should detect some H₀ features (connected components)
        h0_feats = summary.features_by_dim(0)
        assert len(h0_feats) > 0

    def test_high_dim_with_reduction(self, high_dim_embeddings):
        summary = compute_persistence(
            high_dim_embeddings, max_dim=1, reduce_dim=10, reduce_method="pca"
        )
        assert summary.n_points == 100
        assert summary.embedding_dim == 384
        assert summary.reduced_dim == 10
        assert summary.total_features > 0

    def test_to_dict(self, circle_points):
        summary = compute_persistence(circle_points, max_dim=1, reduce_dim=None)
        d = summary.to_dict()
        assert "betti_numbers" in d
        assert "persistence_entropy" in d
        assert "total_features" in d
        assert "top_persistent" in d

    def test_max_dim_0(self, two_clusters):
        summary = compute_persistence(two_clusters, max_dim=0, reduce_dim=None)
        assert summary.max_homology_dim == 0
        assert len(summary.features_by_dim(1)) == 0


class TestReduceEmbeddings:
    def test_pca_reduces(self, high_dim_embeddings):
        reduced = reduce_embeddings(high_dim_embeddings, target_dim=10, method="pca")
        assert reduced.shape == (100, 10)

    def test_no_reduction_if_already_low(self):
        data = np.random.randn(50, 5)
        reduced = reduce_embeddings(data, target_dim=10, method="pca")
        assert reduced.shape == (50, 5)  # unchanged

    def test_umap_reduces(self, high_dim_embeddings):
        reduced = reduce_embeddings(
            high_dim_embeddings, target_dim=5, method="umap", seed=42
        )
        assert reduced.shape == (100, 5)


class TestCompareTopologies:
    def test_same_space(self, circle_points):
        s = compute_persistence(circle_points, max_dim=1, reduce_dim=None)
        comp = compare_topologies(s, s, dim=1)
        assert comp["n_features_a"] == comp["n_features_b"]
        if "wasserstein_distance" in comp:
            assert comp["wasserstein_distance"] == 0.0

    def test_different_spaces(self, circle_points, two_clusters):
        s1 = compute_persistence(circle_points, max_dim=1, reduce_dim=None)
        s2 = compute_persistence(two_clusters, max_dim=1, reduce_dim=None)
        comp = compare_topologies(s1, s2, dim=1)
        assert "n_features_a" in comp
        assert "n_features_b" in comp


class TestPersistenceEntropy:
    def test_uniform(self):
        # Uniform lifetimes → high entropy
        h = _persistence_entropy([1.0, 1.0, 1.0, 1.0])
        assert h > 0.9  # near max

    def test_concentrated(self):
        # One dominant, rest tiny → low entropy
        h = _persistence_entropy([100.0, 0.01, 0.01, 0.01])
        assert h < 0.5

    def test_empty(self):
        assert _persistence_entropy([]) == 0.0

    def test_single(self):
        h = _persistence_entropy([5.0])
        assert h == 0.0  # only one element


class TestTopologicalSummary:
    def test_features_by_dim(self):
        s = TopologicalSummary(
            n_points=10, embedding_dim=2, reduced_dim=None,
            max_homology_dim=1,
            features=[
                PersistenceFeature(0, 0.0, 1.0),
                PersistenceFeature(0, 0.0, 0.5),
                PersistenceFeature(1, 0.3, 1.5),
            ],
            betti_numbers={0: 2, 1: 1},
            persistence_entropy={0: 0.5, 1: 0.0},
        )
        assert len(s.features_by_dim(0)) == 2
        assert len(s.features_by_dim(1)) == 1
        assert s.total_features == 3
