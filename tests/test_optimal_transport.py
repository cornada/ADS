"""Tests for optimal transport module."""
import numpy as np
import pytest

from ads_core.eval.optimal_transport import (
    compute_transport,
    analyze_residuals,
    extract_top_flows,
    compare_distributions,
    cosine_cost_matrix,
    TransportResult,
    ResidualAnalysis,
    OTComparison,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def embeddings(rng):
    src = rng.randn(30, 64).astype(np.float32)
    tgt = rng.randn(20, 64).astype(np.float32)
    return src, tgt


@pytest.fixture
def labels():
    src_labels = [f"course_{i}" for i in range(30)]
    tgt_labels = [f"job_{i}" for i in range(20)]
    return src_labels, tgt_labels


class TestCosineCoostMatrix:
    def test_shape(self, embeddings):
        src, tgt = embeddings
        C = cosine_cost_matrix(src, tgt)
        assert C.shape == (30, 20)

    def test_range(self, embeddings):
        src, tgt = embeddings
        C = cosine_cost_matrix(src, tgt)
        assert C.min() >= -0.01  # numerical tolerance
        assert C.max() <= 2.01

    def test_self_distance_near_zero(self, rng):
        X = rng.randn(10, 32).astype(np.float32)
        C = cosine_cost_matrix(X, X)
        diag = np.diag(C)
        np.testing.assert_allclose(diag, 0, atol=1e-5)

    def test_symmetric(self, rng):
        X = rng.randn(10, 32).astype(np.float32)
        C = cosine_cost_matrix(X, X)
        np.testing.assert_allclose(C, C.T, atol=1e-5)


class TestComputeTransport:
    def test_basic(self, embeddings, labels):
        src, tgt = embeddings
        src_labels, tgt_labels = labels
        result = compute_transport(src, tgt, src_labels, tgt_labels)
        assert isinstance(result, TransportResult)
        assert result.wasserstein_distance > 0
        assert result.transport_plan.shape == (30, 20)

    def test_plan_sums_to_one(self, embeddings, labels):
        src, tgt = embeddings
        src_labels, tgt_labels = labels
        result = compute_transport(src, tgt, src_labels, tgt_labels)
        assert abs(result.transport_plan.sum() - 1.0) < 0.01

    def test_plan_marginals(self, embeddings, labels):
        src, tgt = embeddings
        src_labels, tgt_labels = labels
        result = compute_transport(src, tgt, src_labels, tgt_labels)
        row_sums = result.transport_plan.sum(axis=1)
        col_sums = result.transport_plan.sum(axis=0)
        np.testing.assert_allclose(row_sums, result.source_weights, atol=0.01)
        np.testing.assert_allclose(col_sums, result.target_weights, atol=0.01)

    def test_custom_weights(self, embeddings, labels):
        src, tgt = embeddings
        src_labels, tgt_labels = labels
        w_src = np.ones(30) * 2.0  # will be normalized
        w_tgt = np.ones(20) * 3.0
        result = compute_transport(src, tgt, src_labels, tgt_labels,
                                   source_weights=w_src, target_weights=w_tgt)
        np.testing.assert_allclose(result.source_weights.sum(), 1.0, atol=1e-6)
        np.testing.assert_allclose(result.target_weights.sum(), 1.0, atol=1e-6)

    def test_exact_emd(self, embeddings, labels):
        src, tgt = embeddings
        src_labels, tgt_labels = labels
        result = compute_transport(src, tgt, src_labels, tgt_labels, regularization=0)
        assert result.wasserstein_distance > 0
        assert result.metadata["regularization"] == 0


class TestAnalyzeResiduals:
    def test_basic(self, embeddings, labels):
        src, tgt = embeddings
        src_labels, tgt_labels = labels
        result = compute_transport(src, tgt, src_labels, tgt_labels)
        residuals = analyze_residuals(result)
        assert isinstance(residuals, ResidualAnalysis)
        assert len(residuals.per_source_cost) == 30
        assert len(residuals.per_target_coverage) == 20

    def test_surplus_count(self, embeddings, labels):
        src, tgt = embeddings
        src_labels, tgt_labels = labels
        result = compute_transport(src, tgt, src_labels, tgt_labels)
        residuals = analyze_residuals(result, surplus_threshold=0.1)
        assert len(residuals.surplus_indices) == max(1, int(0.1 * 30))

    def test_costs_non_negative(self, embeddings, labels):
        src, tgt = embeddings
        src_labels, tgt_labels = labels
        result = compute_transport(src, tgt, src_labels, tgt_labels)
        residuals = analyze_residuals(result)
        assert all(c >= 0 for c in residuals.per_source_cost)
        assert all(c >= 0 for c in residuals.per_target_coverage)


class TestExtractTopFlows:
    def test_respects_top_k(self, embeddings, labels):
        src, tgt = embeddings
        src_labels, tgt_labels = labels
        result = compute_transport(src, tgt, src_labels, tgt_labels)
        flows = extract_top_flows(result, top_k=5)
        assert len(flows) <= 5

    def test_flow_structure(self, embeddings, labels):
        src, tgt = embeddings
        src_labels, tgt_labels = labels
        result = compute_transport(src, tgt, src_labels, tgt_labels)
        flows = extract_top_flows(result, top_k=3)
        for f in flows:
            assert "source" in f
            assert "target" in f
            assert "flow_mass" in f
            assert "cost" in f
            assert f["flow_mass"] > 0

    def test_sorted_by_mass(self, embeddings, labels):
        src, tgt = embeddings
        src_labels, tgt_labels = labels
        result = compute_transport(src, tgt, src_labels, tgt_labels)
        flows = extract_top_flows(result, top_k=10)
        masses = [f["flow_mass"] for f in flows]
        assert masses == sorted(masses, reverse=True)


class TestCompareDistributions:
    def test_basic(self, embeddings, labels):
        src, tgt = embeddings
        src_labels, tgt_labels = labels
        comp = compare_distributions("courses", "jobs", src, tgt, src_labels, tgt_labels)
        assert isinstance(comp, OTComparison)
        assert comp.wasserstein > 0
        assert comp.name_a == "courses"
        assert comp.name_b == "jobs"
        assert 0 <= comp.surplus_ratio <= 1
        assert 0 <= comp.deficit_ratio <= 1
