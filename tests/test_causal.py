"""Tests for causal analysis modules."""
import numpy as np
import pandas as pd
import pytest

from ads_core.causal.dag_assumptions import (
    CausalDAG,
    CausalEdge,
    CausalNode,
    NodeType,
    build_ads_causal_dag,
)
from ads_core.causal.sensitivity_analysis import (
    compute_e_value,
    e_value_from_cohens_d,
    rosenbaum_bounds,
    manski_bounds,
)
from ads_core.causal.natural_experiments import (
    detect_fgos_changes,
    diff_in_diff,
    compute_program_metrics,
    run_fgos_did,
    FGOSChangeEvent,
)


# ============================================================================
# DAG Assumptions
# ============================================================================

class TestCausalDAG:
    def test_build_ads_dag(self):
        dag = build_ads_causal_dag()
        assert len(dag.nodes) == 8
        assert len(dag.edges) > 5

    def test_instrument_node(self):
        dag = build_ads_causal_dag()
        assert dag.nodes["fgos_policy"].node_type == NodeType.INSTRUMENT

    def test_unobserved_node(self):
        dag = build_ads_causal_dag()
        assert dag.nodes["student_ability"].node_type == NodeType.UNOBSERVED

    def test_parents(self):
        dag = build_ads_causal_dag()
        parents = dag.parents("curriculum_structure")
        assert "fgos_policy" in parents
        assert "university_resources" in parents

    def test_children(self):
        dag = build_ads_causal_dag()
        children = dag.children("curriculum_structure")
        assert "embedding_position" in children
        assert "competency_coverage" in children

    def test_unobserved_confounders(self):
        dag = build_ads_causal_dag()
        # student_ability confounds competency_coverage → market_fit
        confounders = dag.unobserved_confounders(
            "competency_coverage", "market_fit"
        )
        assert "student_ability" in confounders

    def test_has_path(self):
        dag = build_ads_causal_dag()
        assert dag._has_path_to("fgos_policy", "market_fit")
        assert not dag._has_path_to("market_fit", "fgos_policy")

    def test_to_dict(self):
        dag = build_ads_causal_dag()
        d = dag.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) == 8

    def test_custom_dag(self):
        dag = CausalDAG()
        dag.add_node(CausalNode("A", NodeType.OBSERVED))
        dag.add_node(CausalNode("B", NodeType.OBSERVED))
        dag.add_edge(CausalEdge("A", "B"))
        assert dag.parents("B") == ["A"]
        assert dag.children("A") == ["B"]


# ============================================================================
# Sensitivity Analysis
# ============================================================================

class TestEValue:
    def test_large_rr(self):
        result = compute_e_value(risk_ratio=3.0)
        assert result.e_value > 3.0
        assert "robust" in result.interpretation.lower()

    def test_small_rr(self):
        result = compute_e_value(risk_ratio=1.2)
        assert result.e_value < 2.0
        assert result.e_value > 1.0

    def test_rr_below_one_inverted(self):
        result = compute_e_value(risk_ratio=0.5)
        # Should invert to 2.0
        assert result.point_estimate == 2.0

    def test_with_ci(self):
        result = compute_e_value(risk_ratio=2.5, ci_lower=1.5)
        assert result.e_value_ci > 0
        assert result.e_value_ci < result.e_value

    def test_from_cohens_d(self):
        result = e_value_from_cohens_d(d=0.8)
        assert result.e_value > 1.0
        assert "cohens_d" in result.metadata

    def test_rr_equal_one(self):
        result = compute_e_value(risk_ratio=1.0)
        assert result.e_value == 1.0  # no confounding needed


class TestRosenbaumBounds:
    def test_clear_treatment_effect(self):
        rng = np.random.RandomState(42)
        treated = rng.randn(50) + 2.0  # clear positive shift
        control = rng.randn(50)
        result = rosenbaum_bounds(treated, control)
        assert result.critical_gamma > 1.0
        assert len(result.p_values) == len(result.gamma_values)

    def test_no_effect(self):
        rng = np.random.RandomState(42)
        treated = rng.randn(50)
        control = rng.randn(50)
        result = rosenbaum_bounds(treated, control)
        # With no real effect, even small bias should kill significance
        assert result.critical_gamma <= 2.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            rosenbaum_bounds(np.array([1, 2, 3]), np.array([1, 2]))

    def test_custom_gamma_range(self):
        rng = np.random.RandomState(42)
        treated = rng.randn(30) + 1.0
        control = rng.randn(30)
        result = rosenbaum_bounds(treated, control, gamma_range=[1.0, 1.5, 2.0])
        assert len(result.p_values) == 3


class TestManskiBounds:
    def test_full_observation(self):
        rng = np.random.RandomState(42)
        n = 100
        treatment = np.array([1] * 50 + [0] * 50)
        outcomes = rng.randn(n) + treatment * 1.0  # true effect = 1.0
        selection = np.ones(n)  # all observed
        result = manski_bounds(outcomes, treatment, selection)
        # With full observation, bounds collapse to point estimate
        assert result.lower_bound <= result.upper_bound
        assert abs(result.lower_bound - result.upper_bound) < 0.01
        assert result.selection_fraction == 0.0

    def test_partial_observation(self):
        rng = np.random.RandomState(42)
        n = 100
        treatment = np.array([1] * 50 + [0] * 50)
        outcomes = rng.randn(n) + treatment * 1.0
        selection = np.ones(n)
        selection[80:] = 0  # 20% unobserved
        result = manski_bounds(outcomes, treatment, selection)
        assert result.selection_fraction > 0
        # Bounds should be wider than with full observation
        assert result.upper_bound - result.lower_bound > 0

    def test_empty_group(self):
        result = manski_bounds(
            np.array([1.0, 2.0]),
            np.array([1, 1]),  # all treated, no control
            np.array([1, 1]),
        )
        assert np.isnan(result.point_estimate)


# ============================================================================
# Natural Experiments
# ============================================================================

@pytest.fixture
def misis_like_df():
    """Synthetic DataFrame mimicking MISIS temporal data."""
    rows = []
    # Program A: changes in 2024 (treated)
    for y in [2021, 2022, 2023]:
        for i in range(5):
            rows.append({
                "course_name": f"CourseA_{i}", "fgos_code": "09.04.01",
                "year": y, "credits_zet": 3,
                "competency_codes": "УК-1,ОПК-2",
            })
    for y in [2024, 2025]:
        for i in range(20):  # big jump
            rows.append({
                "course_name": f"CourseA_new_{i}", "fgos_code": "09.04.01",
                "year": y, "credits_zet": 4,
                "competency_codes": "УК-1,ОПК-2,ПК-1,ПК-2",
            })
    # Program B: stable (control)
    for y in [2021, 2022, 2023, 2024, 2025]:
        for i in range(8):
            rows.append({
                "course_name": f"CourseB_{i}", "fgos_code": "38.03.01",
                "year": y, "credits_zet": 3,
                "competency_codes": "УК-1,ОПК-1",
            })
    return pd.DataFrame(rows)


class TestDetectFGOSChanges:
    def test_detects_change(self, misis_like_df):
        events = detect_fgos_changes(misis_like_df)
        assert len(events) > 0
        fgos_changed = {e.fgos_code for e in events}
        assert "09.04.01" in fgos_changed

    def test_no_change_for_stable(self, misis_like_df):
        events = detect_fgos_changes(misis_like_df)
        # Control program should not have events (ratio < 0.5)
        control_events = [e for e in events if e.fgos_code == "38.03.01"]
        assert len(control_events) == 0

    def test_change_year(self, misis_like_df):
        events = detect_fgos_changes(misis_like_df)
        treated = [e for e in events if e.fgos_code == "09.04.01"]
        assert any(e.change_year == 2024 for e in treated)

    def test_pre_post_years(self, misis_like_df):
        events = detect_fgos_changes(misis_like_df)
        for event in events:
            assert len(event.pre_years) > 0
            assert len(event.post_years) > 0


class TestDiffInDiff:
    def test_clear_effect(self):
        rng = np.random.RandomState(42)
        # Treated group jumps by 5, control stays flat
        treated_pre = rng.randn(20) + 10
        treated_post = rng.randn(20) + 15
        control_pre = rng.randn(20) + 10
        control_post = rng.randn(20) + 10
        result = diff_in_diff(treated_pre, treated_post, control_pre, control_post)
        assert result.treatment_effect > 3.0  # should be ~5
        assert result.p_value < 0.05

    def test_no_effect(self):
        rng = np.random.RandomState(42)
        treated_pre = rng.randn(20) + 10
        treated_post = rng.randn(20) + 10
        control_pre = rng.randn(20) + 10
        control_post = rng.randn(20) + 10
        result = diff_in_diff(treated_pre, treated_post, control_pre, control_post)
        assert abs(result.treatment_effect) < 2.0

    def test_to_dict(self):
        result = diff_in_diff(
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
        )
        d = result.to_dict()
        assert "treatment_effect" in d
        assert "ci_95" in d
        assert "interpretation" in d


class TestComputeProgramMetrics:
    def test_basic(self, misis_like_df):
        m = compute_program_metrics(misis_like_df, "09.04.01", 2024)
        assert m["n_courses"] == 20
        assert "mean_credits" in m

    def test_empty(self, misis_like_df):
        m = compute_program_metrics(misis_like_df, "NONEXISTENT", 2024)
        assert m == {}


class TestRunFGOSDiD:
    def test_full_pipeline(self, misis_like_df):
        result = run_fgos_did(misis_like_df, metric="n_courses", treatment_year=2024)
        assert "did_result" in result
        assert result["n_treated_fgos"] > 0
        assert "09.04.01" in result["treated_fgos"]

    def test_no_pre_years(self, misis_like_df):
        result = run_fgos_did(misis_like_df, treatment_year=2021)
        assert "error" in result
