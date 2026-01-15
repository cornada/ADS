"""Tests for explainability and recourse module."""
from __future__ import annotations

import numpy as np
import pytest

from ads_core.explain.explainer import (
    Explainer,
    Explanation,
    RecourseResult,
    explain_option,
    suggest_recourse,
)
from ads_core.lenses.identity import IdentityLens


@pytest.fixture
def sample_data():
    """Create sample data for testing explainability."""
    np.random.seed(42)

    # Option embeddings
    option_embeddings = {
        "course:a": np.array([1.0, 0.0, 0.0, 0.0]),
        "course:b": np.array([0.0, 1.0, 0.0, 0.0]),
        "course:c": np.array([0.5, 0.5, 0.0, 0.0]),
        "course:d": np.array([0.3, 0.3, 0.4, 0.0]),
    }

    # Normalize
    for k in option_embeddings:
        v = option_embeddings[k]
        option_embeddings[k] = v / (np.linalg.norm(v) + 1e-8)

    option_texts = {
        "course:a": "Introduction to Machine Learning",
        "course:b": "AI Ethics and Governance",
        "course:c": "Applied ML Systems",
        "course:d": "Statistical Learning",
    }

    # Target centroids
    target_centroids = {
        "market": np.array([1.0, 0.0, 0.0, 0.0]),
        "mission": np.array([0.0, 1.0, 0.0, 0.0]),
        "learner": np.array([0.5, 0.5, 0.0, 0.0]),
    }
    for k in target_centroids:
        v = target_centroids[k]
        target_centroids[k] = v / (np.linalg.norm(v) + 1e-8)

    # Target artifacts for evidence
    target_artifacts = {
        "market": [
            ("job:ml_eng", "ML Engineer role", np.array([1.0, 0.0, 0.0, 0.0])),
            ("job:data_sci", "Data Scientist role", np.array([0.8, 0.2, 0.0, 0.0])),
        ],
        "mission": [
            ("mission:ethics", "Responsible AI mission", np.array([0.0, 1.0, 0.0, 0.0])),
        ],
        "learner": [
            ("learner:demo", "Want to build ML systems", np.array([0.5, 0.5, 0.0, 0.0])),
        ],
    }

    # Lenses
    lenses = {k: IdentityLens(lens_id=f"identity:{k}") for k in target_centroids}

    # All objectives (pre-computed)
    all_objectives = {
        "course:a": {"market": 1.0, "mission": 0.0, "learner": 0.707},
        "course:b": {"market": 0.0, "mission": 1.0, "learner": 0.707},
        "course:c": {"market": 0.707, "mission": 0.707, "learner": 1.0},
        "course:d": {"market": 0.424, "mission": 0.424, "learner": 0.849},
    }

    pareto_ids = {"course:a", "course:b", "course:c"}  # course:d is dominated

    return {
        "option_embeddings": option_embeddings,
        "option_texts": option_texts,
        "target_centroids": target_centroids,
        "target_artifacts": target_artifacts,
        "lenses": lenses,
        "all_objectives": all_objectives,
        "pareto_ids": pareto_ids,
    }


class TestExplainer:
    """Tests for Explainer class."""

    def test_explain_pareto_option(self, sample_data):
        """Test explaining a Pareto-optimal option."""
        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
        )

        exp = explainer.explain("course:a")

        assert exp.option_id == "course:a"
        assert exp.pareto_status == "pareto"
        assert "market" in exp.objectives
        assert exp.objectives["market"].score == 1.0
        assert exp.objectives["market"].rank == 1
        assert exp.overall_summary is not None

    def test_explain_dominated_option(self, sample_data):
        """Test explaining a dominated option."""
        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
        )

        exp = explainer.explain("course:d")

        assert exp.option_id == "course:d"
        assert exp.pareto_status == "dominated"
        assert "dominated" in exp.pareto_reason.lower() or "dominated" in exp.overall_summary.lower()

    def test_explain_with_evidence(self, sample_data):
        """Test that explanations include evidence."""
        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
        )

        exp = explainer.explain("course:a")

        # Should have evidence for market objective
        assert "market" in exp.objectives
        assert len(exp.objectives["market"].evidence) > 0
        assert exp.objectives["market"].evidence[0].artifact_id.startswith("job:")

    def test_explain_drift(self, sample_data):
        """Test autonomy drift explanation."""
        # Add university objective for drift calculation
        sample_data["all_objectives"]["course:a"]["university"] = 0.5

        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
            autonomy_tau=0.25,
        )

        exp = explainer.explain("course:a")

        assert exp.drift is not None
        assert exp.drift.total_drift >= 0
        assert exp.drift.interpretation is not None

    def test_explain_to_dict(self, sample_data):
        """Test explanation serialization."""
        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
        )

        exp = explainer.explain("course:a")
        d = exp.to_dict()

        assert "option_id" in d
        assert "objectives" in d
        assert "pareto_status" in d
        assert "overall_summary" in d

    def test_unknown_option_raises(self, sample_data):
        """Test that unknown option raises error."""
        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
        )

        with pytest.raises(ValueError):
            explainer.explain("course:unknown")


class TestRecourse:
    """Tests for recourse suggestions."""

    def test_suggest_recourse_finds_alternatives(self, sample_data):
        """Test that recourse finds improving alternatives."""
        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
        )

        # course:a has market=1.0 but mission=0.0
        # Recourse should suggest course:b (mission=1.0) or course:c
        rec = explainer.suggest_recourse("course:a", "mission", min_improvement=0.05)

        assert rec.original_option_id == "course:a"
        assert rec.weak_objective == "mission"
        assert rec.original_score == 0.0
        assert len(rec.alternatives) > 0
        assert rec.best_alternative is not None
        assert rec.best_alternative.improvement > 0

    def test_recourse_includes_tradeoffs(self, sample_data):
        """Test that recourse includes trade-off analysis."""
        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
        )

        rec = explainer.suggest_recourse("course:a", "mission")

        if rec.best_alternative:
            assert "trade_offs" in rec.best_alternative.to_dict()
            # Switching from a to b should show market regression
            if rec.best_alternative.option_id == "course:b":
                assert rec.best_alternative.trade_offs.get("market", 0) < 0

    def test_recourse_no_alternatives(self, sample_data):
        """Test recourse when no improvements exist."""
        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
        )

        # course:a has market=1.0, no improvement possible
        rec = explainer.suggest_recourse("course:a", "market", min_improvement=0.05)

        assert rec.alternatives == []
        assert rec.best_alternative is None
        assert "no alternatives" in rec.summary.lower()

    def test_recourse_to_dict(self, sample_data):
        """Test recourse serialization."""
        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
        )

        rec = explainer.suggest_recourse("course:a", "mission")
        d = rec.to_dict()

        assert "original_option_id" in d
        assert "weak_objective" in d
        assert "alternatives" in d
        assert "summary" in d


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_explain_option_function(self, sample_data):
        """Test standalone explain_option function."""
        exp = explain_option(
            option_id="course:a",
            option_embedding=sample_data["option_embeddings"]["course:a"],
            option_text=sample_data["option_texts"]["course:a"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            objectives=sample_data["all_objectives"]["course:a"],
            pareto_ids=sample_data["pareto_ids"],
        )

        assert isinstance(exp, Explanation)
        assert exp.option_id == "course:a"

    def test_suggest_recourse_function(self, sample_data):
        """Test standalone suggest_recourse function."""
        rec = suggest_recourse(
            option_id="course:a",
            target_objective="mission",
            all_option_embeddings=sample_data["option_embeddings"],
            all_option_texts=sample_data["option_texts"],
            all_objectives=sample_data["all_objectives"],
            target_centroids=sample_data["target_centroids"],
            lenses=sample_data["lenses"],
            pareto_ids=sample_data["pareto_ids"],
        )

        assert isinstance(rec, RecourseResult)
        assert rec.original_option_id == "course:a"
        assert rec.weak_objective == "mission"
