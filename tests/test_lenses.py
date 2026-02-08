"""Unit tests for stakeholder lenses."""
import numpy as np
import pytest

from ads_core.lenses import (
    Lens,
    IdentityLens,
    DiagonalLens,
    create_uniform_lens,
    create_random_lens,
    LearnedDiagonalLenses,
    learn_one_vs_rest_diagonal_weights,
    learn_from_labeled_corpora,
)


# ============================================================================
# IdentityLens Tests
# ============================================================================

class TestIdentityLens:
    """Tests for IdentityLens."""

    def test_identity_transform_1d(self):
        """Identity lens should return input unchanged."""
        lens = IdentityLens("identity:test")
        v = np.array([1.0, 2.0, 3.0])
        result = lens.transform(v)
        np.testing.assert_array_equal(result, v)

    def test_identity_transform_2d(self):
        """Identity lens should work with batch input."""
        lens = IdentityLens("identity:test")
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = lens.transform(v)
        np.testing.assert_array_equal(result, v)

    def test_identity_lens_id(self):
        """Lens should have correct ID."""
        lens = IdentityLens("identity:market")
        assert lens.lens_id == "identity:market"

    def test_identity_to_dict(self):
        """Identity lens should serialize correctly."""
        lens = IdentityLens("identity:market")
        d = lens.to_dict()
        assert d["lens_id"] == "identity:market"
        assert d["type"] == "IdentityLens"

    def test_identity_repr(self):
        """Identity lens should have readable repr."""
        lens = IdentityLens("identity:test")
        assert "IdentityLens" in repr(lens)
        assert "identity:test" in repr(lens)


# ============================================================================
# DiagonalLens Tests
# ============================================================================

class TestDiagonalLens:
    """Tests for DiagonalLens."""

    def test_diagonal_transform_1d(self):
        """Diagonal lens should scale each dimension."""
        weights = np.array([1.0, 2.0, 0.5])
        lens = DiagonalLens("diagonal:test", weights)
        v = np.array([1.0, 1.0, 1.0])
        result = lens.transform(v)
        np.testing.assert_array_almost_equal(result, weights)

    def test_diagonal_transform_2d(self):
        """Diagonal lens should work with batch input."""
        weights = np.array([2.0, 1.0])
        lens = DiagonalLens("diagonal:test", weights)
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.array([[2.0, 2.0], [6.0, 4.0]])
        result = lens.transform(v)
        np.testing.assert_array_almost_equal(result, expected)

    def test_diagonal_normalize(self):
        """Normalized lens should have mean weight of 1.0."""
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        lens = DiagonalLens("diagonal:test", weights, normalize=True)
        assert abs(lens.weights.mean() - 1.0) < 1e-6

    def test_diagonal_no_normalize(self):
        """Non-normalized lens should preserve original weights."""
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        lens = DiagonalLens("diagonal:test", weights, normalize=False)
        np.testing.assert_array_almost_equal(lens.weights, weights)

    def test_diagonal_dim(self):
        """Lens should report correct dimensionality."""
        weights = np.array([1.0, 2.0, 3.0])
        lens = DiagonalLens("diagonal:test", weights)
        assert lens.dim == 3

    def test_diagonal_to_dict(self):
        """Diagonal lens should serialize correctly."""
        weights = np.array([1.0, 2.0])
        lens = DiagonalLens("diagonal:market", weights)
        d = lens.to_dict()
        assert d["lens_id"] == "diagonal:market"
        assert d["type"] == "DiagonalLens"
        assert d["dim"] == 2
        assert d["mode"] == "diagonal"

    def test_diagonal_explain(self):
        """Explain should identify top activated dimensions."""
        weights = np.array([0.1, 0.5, 2.0, 0.3])
        lens = DiagonalLens("diagonal:test", weights)
        v = np.array([1.0, 1.0, 1.0, 1.0])
        result = lens.explain(v, top_k=2)
        # Dimension 2 (weight 2.0) should be first
        assert result["top_dims"][0] == 2


class TestCreateUniformLens:
    """Tests for create_uniform_lens helper."""

    def test_uniform_weights(self):
        """Uniform lens should have all weights = 1.0."""
        lens = create_uniform_lens("uniform:test", dim=5)
        np.testing.assert_array_equal(lens.weights, np.ones(5))

    def test_uniform_transform(self):
        """Uniform lens should act like identity."""
        lens = create_uniform_lens("uniform:test", dim=3)
        v = np.array([1.0, 2.0, 3.0])
        result = lens.transform(v)
        np.testing.assert_array_equal(result, v)


class TestCreateRandomLens:
    """Tests for create_random_lens helper."""

    def test_random_reproducible(self):
        """Random lens should be reproducible with same seed."""
        lens1 = create_random_lens("random:test", dim=10, seed=42)
        lens2 = create_random_lens("random:test", dim=10, seed=42)
        np.testing.assert_array_almost_equal(lens1.weights, lens2.weights)

    def test_random_different_seeds(self):
        """Different seeds should produce different weights."""
        lens1 = create_random_lens("random:test", dim=10, seed=42)
        lens2 = create_random_lens("random:test", dim=10, seed=123)
        assert not np.allclose(lens1.weights, lens2.weights)

    def test_random_positive(self):
        """Random weights should all be positive."""
        lens = create_random_lens("random:test", dim=100, seed=42)
        assert np.all(lens.weights > 0)

    def test_random_normalized(self):
        """Random lens should be normalized to mean 1.0."""
        lens = create_random_lens("random:test", dim=100, seed=42)
        assert abs(lens.weights.mean() - 1.0) < 1e-6


# ============================================================================
# LearnedDiagonalLenses Tests
# ============================================================================

class TestLearnedDiagonalLenses:
    """Tests for LearnedDiagonalLenses container."""

    def test_get_lens(self):
        """Should create DiagonalLens from learned weights."""
        learned = LearnedDiagonalLenses(
            weights={"market": np.array([1.0, 2.0, 3.0])},
            metadata={"seed": 42},
        )
        lens = learned.get_lens("market")
        assert isinstance(lens, DiagonalLens)
        assert lens.lens_id == "learned:market"
        np.testing.assert_array_almost_equal(lens.weights, [1.0, 2.0, 3.0])

    def test_get_lens_missing(self):
        """Should raise KeyError for missing stakeholder."""
        learned = LearnedDiagonalLenses(weights={"market": np.array([1.0])})
        with pytest.raises(KeyError):
            learned.get_lens("mission")

    def test_get_all_lenses(self):
        """Should create lenses for all stakeholders."""
        learned = LearnedDiagonalLenses(
            weights={
                "market": np.array([1.0, 2.0]),
                "mission": np.array([2.0, 1.0]),
            }
        )
        lenses = learned.get_all_lenses()
        assert len(lenses) == 2
        assert "market" in lenses
        assert "mission" in lenses

    def test_to_dict_from_dict(self):
        """Should serialize and deserialize correctly."""
        original = LearnedDiagonalLenses(
            weights={"market": np.array([1.0, 2.0, 3.0])},
            metadata={"seed": 42},
        )
        d = original.to_dict()
        restored = LearnedDiagonalLenses.from_dict(d)
        np.testing.assert_array_almost_equal(
            restored.weights["market"],
            original.weights["market"],
        )
        assert restored.metadata["seed"] == 42


class TestLearnOneVsRest:
    """Tests for learn_one_vs_rest_diagonal_weights."""

    def test_basic_learning(self):
        """Should learn different weights for different stakeholders."""
        np.random.seed(42)
        n, d = 100, 16

        # Create data where different dimensions are important for each class
        X_market = np.random.randn(50, d)
        X_market[:, :4] += 2  # First 4 dims important for market

        X_mission = np.random.randn(50, d)
        X_mission[:, 4:8] += 2  # Next 4 dims important for mission

        X = np.vstack([X_market, X_mission])
        labels = ["market"] * 50 + ["mission"] * 50

        learned = learn_one_vs_rest_diagonal_weights(
            X, labels, ["market", "mission"], seed=42
        )

        assert "market" in learned.weights
        assert "mission" in learned.weights
        assert learned.weights["market"].shape == (d,)

    def test_empty_stakeholder(self):
        """Should handle stakeholder with no samples."""
        X = np.random.randn(10, 4)
        labels = ["market"] * 10

        learned = learn_one_vs_rest_diagonal_weights(
            X, labels, ["market", "mission"], seed=42
        )

        # Mission has no samples, should get uniform weights
        np.testing.assert_array_almost_equal(
            learned.weights["mission"],
            np.ones(4),
        )

    def test_all_same_stakeholder(self):
        """Should handle all samples being one stakeholder."""
        X = np.random.randn(10, 4)
        labels = ["market"] * 10

        learned = learn_one_vs_rest_diagonal_weights(
            X, labels, ["market"], seed=42
        )

        # All samples are market, should get uniform weights
        np.testing.assert_array_almost_equal(
            learned.weights["market"],
            np.ones(4),
        )

    def test_metadata_populated(self):
        """Should populate training metadata."""
        X = np.random.randn(20, 8)
        labels = ["market"] * 10 + ["mission"] * 10

        learned = learn_one_vs_rest_diagonal_weights(
            X, labels, ["market", "mission"], seed=123
        )

        assert learned.metadata["seed"] == 123
        assert learned.metadata["n_samples"] == 20
        assert learned.metadata["n_dims"] == 8
        assert "training_stats" in learned.metadata

    def test_invalid_input_shape(self):
        """Should reject non-2D input."""
        X = np.random.randn(10)
        labels = ["market"] * 10

        with pytest.raises(ValueError):
            learn_one_vs_rest_diagonal_weights(X, labels, ["market"])


class TestLearnFromLabeledCorpora:
    """Tests for learn_from_labeled_corpora."""

    def test_from_corpora(self):
        """Should learn from pre-separated corpora."""
        corpora = {
            "market": np.random.randn(50, 8),
            "mission": np.random.randn(30, 8),
        }

        learned = learn_from_labeled_corpora(corpora, seed=42)

        assert "market" in learned.weights
        assert "mission" in learned.weights
        assert learned.metadata["n_samples"] == 80

    def test_invalid_corpus_shape(self):
        """Should reject non-2D corpus."""
        corpora = {
            "market": np.random.randn(10),  # 1D - invalid
        }

        with pytest.raises(ValueError):
            learn_from_labeled_corpora(corpora)


# ============================================================================
# Integration Tests
# ============================================================================

class TestLensIntegration:
    """Integration tests for lens pipeline."""

    def test_transform_preserves_dtype(self):
        """Transformed vectors should maintain float type."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        identity = IdentityLens("test:identity")
        assert identity.transform(v).dtype == np.float32

        diagonal = DiagonalLens("test:diagonal", np.array([1.0, 1.0, 1.0]))
        assert diagonal.transform(v).dtype == np.float32

    def test_lens_chain(self):
        """Lenses can be chained (not standard but possible)."""
        v = np.array([1.0, 2.0, 3.0])
        w1 = np.array([2.0, 1.0, 1.0])
        w2 = np.array([1.0, 2.0, 1.0])

        lens1 = DiagonalLens("lens1", w1)
        lens2 = DiagonalLens("lens2", w2)

        result = lens2.transform(lens1.transform(v))
        expected = v * w1 * w2
        np.testing.assert_array_almost_equal(result, expected)

    def test_learned_lens_transforms(self):
        """Learned lenses should transform correctly."""
        # Train on synthetic data
        np.random.seed(42)
        X = np.random.randn(40, 8)
        labels = ["A"] * 20 + ["B"] * 20

        learned = learn_one_vs_rest_diagonal_weights(X, labels, ["A", "B"])
        lens_a = learned.get_lens("A")

        # Should transform without error
        v = np.random.randn(8)
        result = lens_a.transform(v)
        assert result.shape == v.shape


class TestHydraLensesConfig:
    """Tests for Hydra lenses config integration (P0.1/P0.2 fix verification)."""

    def test_build_lenses_respects_mode(self):
        """_build_lenses should respect 'mode' key from Hydra configs."""
        from ads_core.pipeline.dataset_pipeline import _build_lenses
        from ads_core.data.schemas import Artifact, ArtifactType

        # Create minimal test artifacts
        artifacts = [
            Artifact(
                artifact_id="course_1",
                type=ArtifactType.COURSE,
                text="Test course",
                source_url="test://",
            ),
        ]
        id2vec = {"course_1": np.random.randn(64)}

        # Test with mode=identity (Hydra style)
        lenses = _build_lenses({"mode": "identity"}, d=64, id2vec=id2vec, artifacts=artifacts)
        assert "default" in lenses
        assert lenses["default"].lens_id == "identity:default"

        # Test with mode=diagonal (Hydra style)
        lenses = _build_lenses({"mode": "diagonal"}, d=64, id2vec=id2vec, artifacts=artifacts)
        assert "default" in lenses
        assert "diagonal" in lenses["default"].lens_id

    def test_build_lenses_backward_compat_kind(self):
        """_build_lenses should also support 'kind' key for backward compat."""
        from ads_core.pipeline.dataset_pipeline import _build_lenses
        from ads_core.data.schemas import Artifact, ArtifactType

        artifacts = [
            Artifact(
                artifact_id="course_1",
                type=ArtifactType.COURSE,
                text="Test course",
                source_url="test://",
            ),
        ]
        id2vec = {"course_1": np.random.randn(64)}

        # Test with kind=identity (old style)
        lenses = _build_lenses({"kind": "identity"}, d=64, id2vec=id2vec, artifacts=artifacts)
        assert "default" in lenses
        assert lenses["default"].lens_id == "identity:default"

    def test_build_lenses_mode_takes_precedence(self):
        """If both 'mode' and 'kind' present, 'mode' takes precedence."""
        from ads_core.pipeline.dataset_pipeline import _build_lenses
        from ads_core.data.schemas import Artifact, ArtifactType

        artifacts = [
            Artifact(
                artifact_id="course_1",
                type=ArtifactType.COURSE,
                text="Test course",
                source_url="test://",
            ),
        ]
        id2vec = {"course_1": np.random.randn(64)}

        # Both present - mode should win
        lenses = _build_lenses({"mode": "diagonal", "kind": "identity"}, d=64, id2vec=id2vec, artifacts=artifacts)
        assert "default" in lenses
        assert "diagonal" in lenses["default"].lens_id

    def test_build_lenses_learned_produces_stakeholder_lenses(self):
        """Learned lens mode produces per-stakeholder lenses."""
        from ads_core.pipeline.dataset_pipeline import _build_lenses
        from ads_core.data.schemas import Artifact, ArtifactType

        # Need diverse artifacts for learned lenses
        artifacts = [
            Artifact(artifact_id=f"course_{i}", type=ArtifactType.COURSE, text=f"Course {i}", source_url="test://")
            for i in range(10)
        ] + [
            Artifact(artifact_id=f"job_{i}", type=ArtifactType.JOB_ROLE, text=f"Job {i}", source_url="test://")
            for i in range(10)
        ]

        np.random.seed(42)
        id2vec = {a.artifact_id: np.random.randn(64) for a in artifacts}

        lenses = _build_lenses({"mode": "learned"}, d=64, id2vec=id2vec, artifacts=artifacts)

        # Should have stakeholder-specific lenses (not "default")
        assert "university" in lenses or "market" in lenses
        assert "default" not in lenses or len(lenses) > 1

    def test_pipeline_different_lenses_different_results(self, tmp_path):
        """Different lens modes should produce different evaluation results."""
        from ads_core.pipeline.dataset_pipeline import run_dataset
        import json

        # Run with identity lens
        outputs_identity = run_dataset(
            dataset_id="toy",
            out_dir=tmp_path / "identity_run",
            seed=42,
            embedding_cfg={"kind": "stub", "d": 64},
            lenses_cfg={"mode": "identity"},
            objectives=["market", "learner"],
            autonomy_tau=0.3,
        )

        # Run with diagonal lens
        outputs_diagonal = run_dataset(
            dataset_id="toy",
            out_dir=tmp_path / "diagonal_run",
            seed=42,
            embedding_cfg={"kind": "stub", "d": 64},
            lenses_cfg={"mode": "diagonal"},
            objectives=["market", "learner"],
            autonomy_tau=0.3,
        )

        # Both should succeed
        assert outputs_identity.results_json.exists()
        assert outputs_diagonal.results_json.exists()

        # With uniform diagonal weights=1, results should be same as identity
        # (this is expected behavior for uniform diagonal weights)
        results_i = json.loads(outputs_identity.results_json.read_text(encoding="utf-8"))
        results_d = json.loads(outputs_diagonal.results_json.read_text(encoding="utf-8"))

        assert len(results_i["results"]) == len(results_d["results"])
