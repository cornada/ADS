"""Tests for objective functions and target sets."""
from __future__ import annotations

import numpy as np
import pytest

from ads_core.eval.distances import cosine_distance, cosine_similarity
from ads_core.eval.objectives import (
    centroid,
    evaluate_option,
    EvaluationResult,
    ObjectiveEvaluator,
    ObjectiveResult,
    normalize_objectives,
    objectives_to_csv_rows,
)
from ads_core.eval.targets import (
    AggregationMethod,
    CentroidTarget,
    CloudTarget,
    TargetSetCollection,
    build_centroid_target,
    build_cloud_target,
    build_single_point_target,
)
from ads_core.lenses.identity import IdentityLens


class TestDistances:
    """Tests for distance functions."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        v = np.array([1.0, 0.0, 0.0])
        sim = cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        sim = cosine_similarity(v1, v2)
        assert abs(sim - 0.0) < 1e-6

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        sim = cosine_similarity(v1, v2)
        assert abs(sim - (-1.0)) < 1e-6

    def test_cosine_distance_range(self):
        """Test cosine distance is in [0, 2]."""
        v1 = np.random.randn(10)
        v2 = np.random.randn(10)
        dist = cosine_distance(v1, v2)
        assert 0.0 <= dist <= 2.0

    def test_similarity_distance_relationship(self):
        """Test that similarity + distance = 1 for normalized vectors."""
        v1 = np.array([0.6, 0.8, 0.0])
        v2 = np.array([0.8, 0.6, 0.0])
        sim = cosine_similarity(v1, v2)
        dist = cosine_distance(v1, v2)
        assert abs((1 - dist) - sim) < 1e-6


class TestCentroid:
    """Tests for centroid computation."""

    def test_centroid_single_vector(self):
        """Test centroid of single vector is normalized."""
        v = np.array([3.0, 4.0, 0.0])
        c = centroid([v])
        norm = np.linalg.norm(c)
        assert abs(norm - 1.0) < 1e-6

    def test_centroid_multiple_vectors(self):
        """Test centroid of multiple vectors."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        c = centroid([v1, v2])
        # Mean is [0.5, 0.5], normalized
        expected = np.array([0.5, 0.5]) / np.linalg.norm([0.5, 0.5])
        np.testing.assert_array_almost_equal(c, expected)

    def test_centroid_empty_raises(self):
        """Test centroid raises on empty input."""
        with pytest.raises(ValueError, match="Empty centroid"):
            centroid([])


class TestCentroidTarget:
    """Tests for CentroidTarget."""

    def test_centroid_target_distance(self):
        """Test centroid target distance computation."""
        c = np.array([1.0, 0.0, 0.0])
        target = CentroidTarget(_name="test", centroid=c, source_count=1)

        v_same = np.array([1.0, 0.0, 0.0])
        v_orth = np.array([0.0, 1.0, 0.0])

        assert target.distance(v_same) < 0.01  # Very close
        assert abs(target.distance(v_orth) - 1.0) < 0.01  # Orthogonal

    def test_centroid_target_similarity(self):
        """Test centroid target similarity computation."""
        c = np.array([1.0, 0.0, 0.0])
        target = CentroidTarget(_name="test", centroid=c, source_count=1)

        v_same = np.array([1.0, 0.0, 0.0])
        assert target.similarity(v_same) > 0.99

    def test_centroid_target_to_dict(self):
        """Test centroid target serialization."""
        c = np.array([1.0, 0.0])
        target = CentroidTarget(_name="market", centroid=c, source_count=10)
        d = target.to_dict()
        assert d["name"] == "market"
        assert d["type"] == "centroid"
        assert d["source_count"] == 10


class TestCloudTarget:
    """Tests for CloudTarget."""

    def test_cloud_target_min_aggregation(self):
        """Test cloud target with min aggregation (closest point)."""
        points = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        target = CloudTarget(
            _name="test",
            points=points,
            aggregation=AggregationMethod.MIN,
        )

        v = np.array([0.9, 0.1])  # Close to first point
        dist = target.distance(v)
        # Should be close to distance to [1, 0]
        assert dist < 0.5

    def test_cloud_target_mean_aggregation(self):
        """Test cloud target with mean aggregation."""
        points = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
        ])
        target = CloudTarget(
            _name="test",
            points=points,
            aggregation=AggregationMethod.MEAN,
        )

        v = np.array([1.0, 0.0])
        dist = target.distance(v)
        # Mean of distance to [1,0] (0) and [-1,0] (2) = 1
        assert abs(dist - 1.0) < 0.1

    def test_cloud_target_empty(self):
        """Test cloud target with no points."""
        target = CloudTarget(
            _name="test",
            points=np.zeros((0, 2)),
            aggregation=AggregationMethod.MIN,
        )
        v = np.array([1.0, 0.0])
        assert target.distance(v) == 2.0  # Max distance


class TestTargetSetBuilders:
    """Tests for target set builder functions."""

    def test_build_centroid_target(self):
        """Test building centroid target from vectors."""
        vectors = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        ]
        target = build_centroid_target("test", vectors)

        assert target.name == "test"
        assert target.source_count == 2
        assert np.linalg.norm(target.centroid) - 1.0 < 1e-6

    def test_build_cloud_target(self):
        """Test building cloud target from vectors."""
        vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        target = build_cloud_target("test", vectors, AggregationMethod.MIN)

        assert target.name == "test"
        assert target.size == 2

    def test_build_single_point_target(self):
        """Test building single point target."""
        v = np.array([3.0, 4.0])
        target = build_single_point_target("learner", v)

        assert target.name == "learner"
        assert target.source_count == 1
        assert np.linalg.norm(target.centroid) - 1.0 < 1e-6


class TestTargetSetCollection:
    """Tests for TargetSetCollection."""

    def test_collection_add_and_get(self):
        """Test adding and retrieving targets."""
        collection = TargetSetCollection()

        t1 = build_centroid_target("market", [np.array([1.0, 0.0])])
        t2 = build_centroid_target("mission", [np.array([0.0, 1.0])])

        collection.add(t1)
        collection.add(t2)

        assert collection.get("market") is t1
        assert collection.get("mission") is t2
        assert set(collection.names()) == {"market", "mission"}

    def test_collection_evaluate(self):
        """Test evaluating distances across collection."""
        collection = TargetSetCollection()
        collection.add(build_centroid_target("a", [np.array([1.0, 0.0])]))
        collection.add(build_centroid_target("b", [np.array([0.0, 1.0])]))

        v = np.array([1.0, 0.0])
        distances = collection.evaluate_distances(v)

        assert "a" in distances
        assert "b" in distances
        assert distances["a"] < distances["b"]  # v is closer to a


class TestEvaluateOption:
    """Tests for evaluate_option function."""

    def test_evaluate_option_identity_lens(self):
        """Test evaluation with identity lens."""
        targets = {
            "market": np.array([1.0, 0.0]),
            "mission": np.array([0.0, 1.0]),
        }
        lenses = {
            "market": IdentityLens(lens_id="id:market"),
            "mission": IdentityLens(lens_id="id:mission"),
        }

        option_v = np.array([0.8, 0.6])
        scores = evaluate_option(option_v, targets, lenses)

        assert "market" in scores
        assert "mission" in scores
        # Scores are similarities (higher is better)
        assert 0 <= scores["market"] <= 1
        assert 0 <= scores["mission"] <= 1


class TestObjectiveEvaluator:
    """Tests for ObjectiveEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create test evaluator."""
        targets = {
            "market": np.array([1.0, 0.0, 0.0]),
            "mission": np.array([0.0, 1.0, 0.0]),
            "learner": np.array([0.0, 0.0, 1.0]),
        }
        lenses = {k: IdentityLens(lens_id=f"id:{k}") for k in targets}
        return ObjectiveEvaluator(targets=targets, lenses=lenses)

    def test_evaluate_single(self, evaluator):
        """Test evaluating single option."""
        option_v = np.array([0.6, 0.6, 0.6])
        option_v = option_v / np.linalg.norm(option_v)

        result = evaluator.evaluate("course:test", option_v)

        assert result.option_id == "course:test"
        assert len(result.objectives) == 3
        assert all(0 <= v <= 1 for v in result.objectives.values())

    def test_evaluate_batch(self, evaluator):
        """Test batch evaluation."""
        options = {
            "opt1": np.array([1.0, 0.0, 0.0]),
            "opt2": np.array([0.0, 1.0, 0.0]),
        }
        results = evaluator.evaluate_batch(options)

        assert len(results) == 2
        assert results[0].option_id == "opt1"
        assert results[1].option_id == "opt2"

    def test_evaluate_specific_objectives(self, evaluator):
        """Test evaluation with specific objectives only."""
        option_v = np.array([1.0, 0.0, 0.0])
        result = evaluator.evaluate("test", option_v, objective_names=["market"])

        assert "market" in result.objectives
        assert "mission" not in result.objectives


class TestEvaluationResult:
    """Tests for EvaluationResult."""

    def test_to_dict(self):
        """Test result serialization."""
        result = EvaluationResult(
            option_id="course:dl",
            objectives={"market": 0.85, "mission": 0.72},
            distances={"market": 0.15, "mission": 0.28},
            feasible=True,
        )
        d = result.to_dict()

        assert d["option_id"] == "course:dl"
        assert d["feasible"] is True
        assert "objectives" in d

    def test_to_row(self):
        """Test CSV row export."""
        result = EvaluationResult(
            option_id="course:dl",
            objectives={"market": 0.85, "mission": 0.72},
            distances={},
            feasible=True,
        )
        row = result.to_row(keys=["market", "mission"])

        assert row["option_id"] == "course:dl"
        assert row["feasible"] is True
        assert row["market"] == 0.85
        assert row["mission"] == 0.72


class TestNormalization:
    """Tests for objective normalization."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        results = [
            EvaluationResult("a", {"x": 0.2}, {}),
            EvaluationResult("b", {"x": 0.8}, {}),
            EvaluationResult("c", {"x": 0.5}, {}),
        ]
        normalized = normalize_objectives(results, method="minmax")

        # After minmax: a=0, b=1, c=0.5
        assert abs(normalized[0].objectives["x"] - 0.0) < 1e-6
        assert abs(normalized[1].objectives["x"] - 1.0) < 1e-6
        assert abs(normalized[2].objectives["x"] - 0.5) < 1e-6

    def test_normalization_preserves_metadata(self):
        """Test normalization preserves other fields."""
        results = [
            EvaluationResult("a", {"x": 0.5}, {}, feasible=False),
        ]
        normalized = normalize_objectives(results, method="minmax")

        assert normalized[0].option_id == "a"
        assert normalized[0].feasible is False


class TestCsvExport:
    """Tests for CSV export utilities."""

    def test_objectives_to_csv_rows(self):
        """Test converting results to CSV rows."""
        results = [
            EvaluationResult("a", {"x": 0.5, "y": 0.6}, {}, feasible=True),
            EvaluationResult("b", {"x": 0.7, "y": 0.3}, {}, feasible=False),
        ]
        rows = objectives_to_csv_rows(results, ["x", "y"])

        assert len(rows) == 2
        assert rows[0]["option_id"] == "a"
        assert rows[0]["x"] == 0.5
        assert rows[1]["feasible"] is False
