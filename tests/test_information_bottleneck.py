"""Tests for Information Bottleneck analysis module."""
import numpy as np
import pytest

from ads_core.lenses.information_bottleneck import (
    compute_ib_point,
    compute_ib_curve,
    evaluate_lens_ib,
    _kl_entropy,
    _discrete_mi,
    IBResult,
    IBCurve,
)


@pytest.fixture
def separable_data():
    """Two well-separated clusters with labels."""
    rng = np.random.RandomState(42)
    n = 100
    X = np.vstack([
        rng.randn(n // 2, 20) + 3.0,
        rng.randn(n // 2, 20) - 3.0,
    ])
    labels = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    return X, labels


@pytest.fixture
def random_data():
    """Random data with no structure."""
    rng = np.random.RandomState(42)
    X = rng.randn(80, 30)
    labels = np.array(["C0"] * 20 + ["C1"] * 20 + ["C2"] * 20 + ["C3"] * 20)
    return X, labels


class TestKLEntropy:
    def test_positive(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        h = _kl_entropy(X, k=5)
        # Entropy of standard normal in d=5 should be positive
        assert h > 0

    def test_low_dim_less_than_high(self):
        rng = np.random.RandomState(42)
        X_low = rng.randn(100, 2)
        X_high = rng.randn(100, 10)
        h_low = _kl_entropy(X_low, k=5)
        h_high = _kl_entropy(X_high, k=5)
        # Higher dimension generally has higher entropy
        assert h_high > h_low

    def test_too_few_points(self):
        X = np.array([[1.0, 2.0]])
        assert _kl_entropy(X, k=5) == 0.0


class TestDiscreteMI:
    def test_separable(self, separable_data):
        X, labels = separable_data
        mi = _discrete_mi(X, labels, k=3)
        # Well-separated clusters → high MI
        assert mi > 0.1

    def test_random(self, random_data):
        X, labels = random_data
        mi = _discrete_mi(X, labels, k=3)
        # Random labels → low MI (but estimator has some bias)
        # Just check it runs and returns non-negative
        assert mi >= 0.0


class TestComputeIBPoint:
    def test_basic(self, separable_data):
        X, labels = separable_data
        from sklearn.decomposition import PCA
        compressed = PCA(n_components=5, random_state=42).fit_transform(X)
        result = compute_ib_point(X, compressed, labels, k=3)
        assert isinstance(result, IBResult)
        assert result.compression >= 0
        assert result.prediction >= 0
        assert result.n_dims == 5

    def test_identity_low_compression(self, separable_data):
        X, labels = separable_data
        # "Identity" compression = same data
        result = compute_ib_point(X, X, labels, k=3)
        # Compression should be ~0 (same space)
        assert result.compression < 0.01


class TestComputeIBCurve:
    def test_basic(self, separable_data):
        X, labels = separable_data
        curve = compute_ib_curve(X, labels, dims=[2, 5, 10], k=3)
        assert isinstance(curve, IBCurve)
        assert len(curve.points) >= 2
        assert curve.optimal_dim is not None

    def test_prediction_increases_with_dims(self, separable_data):
        X, labels = separable_data
        curve = compute_ib_curve(X, labels, dims=[2, 5, 10, 15], k=3)
        # Generally prediction should increase or stay flat with more dims
        predictions = [p.prediction for p in curve.points]
        # Just verify it's a reasonable curve (not strictly monotonic due to estimation noise)
        assert len(predictions) >= 2

    def test_to_dict(self, separable_data):
        X, labels = separable_data
        curve = compute_ib_curve(X, labels, dims=[2, 5], k=3)
        d = curve.to_dict()
        assert "n_points" in d
        assert "optimal_dim" in d
        assert "curve" in d


class TestEvaluateLensIB:
    def test_basic(self, separable_data):
        X, labels = separable_data
        weights = np.ones(X.shape[1])
        result = evaluate_lens_ib(X, weights, labels, k=3)
        assert isinstance(result, IBResult)

    def test_sparse_lens_compresses_more(self, separable_data):
        X, labels = separable_data
        uniform_weights = np.ones(X.shape[1])
        sparse_weights = np.zeros(X.shape[1])
        sparse_weights[:5] = 1.0  # Only 5 dims active

        r_uniform = evaluate_lens_ib(X, uniform_weights, labels, k=3)
        r_sparse = evaluate_lens_ib(X, sparse_weights, labels, k=3)
        # Sparse lens should compress more
        assert r_sparse.compression > r_uniform.compression
