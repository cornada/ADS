"""Tests for explainability and recourse module."""
from __future__ import annotations

import numpy as np
import pytest

from ads_core.explain.explainer import (
    DEFAULT_EVIDENCE_THRESHOLD,
    Explainer,
    Explanation,
    RecourseResult,
    explain_option,
    suggest_recourse,
)
from ads_core.eval.pareto import dominates
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


class TestDominanceCorrectness:
    """Tests for dominance correctness (KT18).

    Ensures explanations use canonical dominates() and handle edge cases.
    """

    def test_identical_objectives_neither_dominates(self):
        """Test that identical objective vectors result in neither dominating.

        This is a critical edge case: if A == B on all objectives,
        then neither A dominates B nor B dominates A.
        """
        a = {"market": 0.5, "mission": 0.5, "learner": 0.5}
        b = {"market": 0.5, "mission": 0.5, "learner": 0.5}
        keys = ["market", "mission", "learner"]

        # Neither should dominate the other
        assert not dominates(a, b, keys)
        assert not dominates(b, a, keys)

    def test_strict_dominance_requires_one_better(self):
        """Test that dominance requires at least one strictly better objective."""
        a = {"market": 0.6, "mission": 0.5, "learner": 0.5}
        b = {"market": 0.5, "mission": 0.5, "learner": 0.5}
        keys = ["market", "mission", "learner"]

        # A should dominate B (equal on 2, strictly better on 1)
        assert dominates(a, b, keys)
        assert not dominates(b, a, keys)

    def test_weak_dominance_not_counted(self):
        """Test that being equal on all but worse on one is not dominating."""
        a = {"market": 0.5, "mission": 0.5, "learner": 0.5}
        b = {"market": 0.5, "mission": 0.5, "learner": 0.4}  # worse on learner
        keys = ["market", "mission", "learner"]

        # A dominates B (equal on 2, strictly better on learner)
        assert dominates(a, b, keys)
        # B does NOT dominate A
        assert not dominates(b, a, keys)

    def test_incomparable_solutions(self):
        """Test that trade-off solutions are incomparable (neither dominates)."""
        a = {"market": 0.8, "mission": 0.3}  # high market, low mission
        b = {"market": 0.3, "mission": 0.8}  # low market, high mission
        keys = ["market", "mission"]

        # Neither dominates - they represent different trade-offs
        assert not dominates(a, b, keys)
        assert not dominates(b, a, keys)

    def test_explainer_uses_canonical_dominates(self, sample_data):
        """Test that explainer's dominance check matches pareto.dominates()."""
        # Add a new option that is identical to course:c
        sample_data["option_embeddings"]["course:e"] = sample_data["option_embeddings"]["course:c"].copy()
        sample_data["option_texts"]["course:e"] = "Duplicate of course:c"
        sample_data["all_objectives"]["course:e"] = sample_data["all_objectives"]["course:c"].copy()
        sample_data["pareto_ids"].add("course:e")  # Both should be Pareto

        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
        )

        # Both c and e should show as Pareto (neither dominates the other)
        exp_c = explainer.explain("course:c")
        exp_e = explainer.explain("course:e")

        assert exp_c.pareto_status == "pareto"
        assert exp_e.pareto_status == "pareto"

    def test_dominated_explanation_shows_dominating_objectives(self, sample_data):
        """Test that dominated explanation shows which objectives caused it."""
        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
        )

        # course:d is dominated by course:c
        exp = explainer.explain("course:d")

        assert exp.pareto_status == "dominated"
        # Reason should mention which option dominates and which objectives
        assert "course:c" in exp.pareto_reason
        assert "strictly better" in exp.pareto_reason.lower() or "better" in exp.pareto_reason.lower()


class TestEvidenceThreshold:
    """Tests for configurable evidence threshold (KT18)."""

    def test_default_evidence_threshold(self):
        """Test that default evidence threshold is 0.5."""
        assert DEFAULT_EVIDENCE_THRESHOLD == 0.5

    def test_custom_evidence_threshold(self, sample_data):
        """Test that custom evidence threshold is used."""
        # Use a very high threshold
        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
            evidence_threshold=0.99,  # Very high threshold
        )

        assert explainer.evidence_threshold == 0.99

        # With high threshold, most evidence should be "weakens"
        exp = explainer.explain("course:c")
        for obj_name, obj_exp in exp.objectives.items():
            for evidence in obj_exp.evidence:
                # Most should be "weakens" with such high threshold
                if evidence.similarity < 0.99:
                    assert evidence.contribution == "weakens"

    def test_low_evidence_threshold_more_supports(self, sample_data):
        """Test that lower threshold results in more 'supports' contributions."""
        # Use a very low threshold
        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
            evidence_threshold=0.1,  # Very low threshold
        )

        assert explainer.evidence_threshold == 0.1

        # With low threshold, most evidence should be "supports"
        exp = explainer.explain("course:a")
        market_evidence = exp.objectives["market"].evidence
        supports_count = sum(1 for e in market_evidence if e.contribution == "supports")
        # High similarity with market artifacts should be "supports"
        assert supports_count > 0


class TestExplanationDeterminism:
    """Tests that explanations are deterministic and consistent."""

    def test_explanation_deterministic(self, sample_data):
        """Test that same inputs produce identical explanations."""
        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=sample_data["pareto_ids"],
        )

        exp1 = explainer.explain("course:a")
        exp2 = explainer.explain("course:a")

        # Should be identical
        assert exp1.pareto_status == exp2.pareto_status
        assert exp1.pareto_reason == exp2.pareto_reason
        assert exp1.overall_summary == exp2.overall_summary

        # Objective scores should match
        for obj in exp1.objectives:
            assert exp1.objectives[obj].score == exp2.objectives[obj].score
            assert exp1.objectives[obj].rank == exp2.objectives[obj].rank

    def test_explanation_consistent_with_pareto_engine(self, sample_data):
        """Test that explanation status matches Pareto engine classification."""
        from ads_core.eval.pareto import pareto_front

        # Compute Pareto front from objectives
        items = [
            sample_data["all_objectives"]["course:a"],
            sample_data["all_objectives"]["course:b"],
            sample_data["all_objectives"]["course:c"],
            sample_data["all_objectives"]["course:d"],
        ]
        keys = ["market", "mission", "learner"]
        pareto_indices = pareto_front(items, keys)

        option_ids = ["course:a", "course:b", "course:c", "course:d"]
        pareto_ids_from_engine = {option_ids[i] for i in pareto_indices}

        explainer = Explainer(
            option_embeddings=sample_data["option_embeddings"],
            option_texts=sample_data["option_texts"],
            target_centroids=sample_data["target_centroids"],
            target_artifacts=sample_data["target_artifacts"],
            lenses=sample_data["lenses"],
            all_objectives=sample_data["all_objectives"],
            pareto_ids=pareto_ids_from_engine,  # Use engine-computed Pareto
        )

        # Explanations should be consistent with engine
        for oid in option_ids:
            exp = explainer.explain(oid)
            if oid in pareto_ids_from_engine:
                assert exp.pareto_status == "pareto", f"{oid} should be pareto"
            else:
                assert exp.pareto_status in ("dominated", "infeasible"), f"{oid} should not be pareto"
