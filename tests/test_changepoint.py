"""Tests for change-point detection module."""
import numpy as np
import pandas as pd
import pytest

from ads_core.temporal.changepoint import (
    build_program_signals,
    detect_changepoints,
    detect_program_changepoints,
    permutation_test_changepoint,
    aggregate_changepoints,
    ChangePoint,
    ProgramEvolution,
)


@pytest.fixture
def simple_program_df():
    """Synthetic program with clear regime shift at 2023."""
    rows = []
    # 2021-2022: small program, 3 courses each
    for year in [2021, 2022]:
        for i in range(3):
            rows.append({
                "course_name": f"Course_{chr(65+i)}",
                "fgos_code": "09.04.01",
                "year": year,
                "credits_zet": 3,
                "competency_codes": "ОПК-1,ОПК-2",
                "content_summary": f"Content {i}",
            })
    # 2023-2024: expanded program, 10 courses, different competencies
    for year in [2023, 2024]:
        for i in range(10):
            rows.append({
                "course_name": f"NewCourse_{i}",
                "fgos_code": "09.04.01",
                "year": year,
                "credits_zet": 5,
                "competency_codes": "ОПК-1,ОПК-2,ОПК-3,ПК-1,ПК-2",
                "content_summary": f"New content {i}",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def stable_program_df():
    """Program with no significant change."""
    rows = []
    for year in [2021, 2022, 2023, 2024]:
        for i in range(5):
            rows.append({
                "course_name": f"Course_{chr(65+i)}",
                "fgos_code": "22.04.01",
                "year": year,
                "credits_zet": 4,
                "competency_codes": "ОПК-1,ОПК-2,ОПК-3",
                "content_summary": f"Content {i}",
            })
    return pd.DataFrame(rows)


class TestBuildProgramSignals:
    def test_basic_signals(self, simple_program_df):
        evo = build_program_signals(simple_program_df, "09.04.01")
        assert evo.fgos_code == "09.04.01"
        assert evo.n_years == 4
        assert "n_courses" in evo.signals
        assert "mean_credits" in evo.signals
        assert "competency_jaccard" in evo.signals
        assert "course_churn" in evo.signals

    def test_n_courses_signal(self, simple_program_df):
        evo = build_program_signals(simple_program_df, "09.04.01")
        n_courses = evo.signals["n_courses"]
        assert n_courses[0] == 3.0  # 2021
        assert n_courses[1] == 3.0  # 2022
        assert n_courses[2] == 10.0  # 2023
        assert n_courses[3] == 10.0  # 2024

    def test_mean_credits_signal(self, simple_program_df):
        evo = build_program_signals(simple_program_df, "09.04.01")
        credits = evo.signals["mean_credits"]
        assert credits[0] == 3.0  # 2021
        assert credits[2] == 5.0  # 2023

    def test_insufficient_years(self):
        df = pd.DataFrame({
            "course_name": ["A"],
            "fgos_code": ["99.99.99"],
            "year": [2024],
            "credits_zet": [3],
            "competency_codes": ["ОПК-1"],
            "content_summary": ["test"],
        })
        evo = build_program_signals(df, "99.99.99")
        assert evo.n_years == 1
        assert evo.signals == {}

    def test_nonexistent_fgos(self, simple_program_df):
        evo = build_program_signals(simple_program_df, "00.00.00")
        assert evo.n_years == 0


class TestDetectChangepoints:
    def test_clear_step_signal(self):
        signal = [1.0, 1.0, 1.0, 10.0, 10.0]
        years = [2020, 2021, 2022, 2023, 2024]
        cps = detect_changepoints(signal, years, signal_name="test")
        assert len(cps) >= 1
        # Should detect shift between 2022 and 2023
        change_years = {(cp.year_before, cp.year_after) for cp in cps}
        assert (2022, 2023) in change_years

    def test_constant_signal_no_cp(self):
        signal = [5.0, 5.0, 5.0, 5.0, 5.0]
        years = [2020, 2021, 2022, 2023, 2024]
        cps = detect_changepoints(signal, years)
        assert len(cps) == 0

    def test_too_short_signal(self):
        signal = [1.0, 2.0]
        years = [2020, 2021]
        cps = detect_changepoints(signal, years)
        assert len(cps) == 0

    def test_empty_signal(self):
        cps = detect_changepoints([], [])
        assert len(cps) == 0

    def test_magnitude_filter(self):
        # Small fluctuation should be filtered out (< 20% of range)
        signal = [10.0, 10.5, 10.0, 10.2, 10.0]
        years = [2020, 2021, 2022, 2023, 2024]
        cps = detect_changepoints(signal, years)
        # With magnitude filter, small fluctuations should produce 0 or few CPs
        for cp in cps:
            signal_range = max(signal) - min(signal)
            assert cp.magnitude >= 0.2 * signal_range

    def test_changepoint_properties(self):
        signal = [0.0, 0.0, 5.0, 5.0, 5.0]
        years = [2020, 2021, 2022, 2023, 2024]
        cps = detect_changepoints(signal, years, signal_name="credits")
        if cps:
            cp = cps[0]
            assert cp.signal_name == "credits"
            assert cp.magnitude > 0
            assert cp.year_before < cp.year_after


class TestDetectProgramChangepoints:
    def test_with_regime_shift(self, simple_program_df):
        evo = build_program_signals(simple_program_df, "09.04.01")
        evo = detect_program_changepoints(evo)
        assert evo.has_changepoints
        # n_courses should show a shift (3 → 10)
        n_course_cps = [cp for cp in evo.changepoints if cp.signal_name == "n_courses"]
        assert len(n_course_cps) >= 1

    def test_stable_program(self, stable_program_df):
        evo = build_program_signals(stable_program_df, "22.04.01")
        evo = detect_program_changepoints(evo)
        # Stable program: n_courses constant (5), credits constant (4)
        n_course_cps = [cp for cp in evo.changepoints if cp.signal_name == "n_courses"]
        assert len(n_course_cps) == 0

    def test_signal_subset(self, simple_program_df):
        evo = build_program_signals(simple_program_df, "09.04.01")
        evo = detect_program_changepoints(evo, signals_to_check=["n_courses"])
        for cp in evo.changepoints:
            assert cp.signal_name == "n_courses"


class TestPermutationTest:
    def test_basic(self):
        signal = [1.0, 1.0, 1.0, 100.0, 100.0]
        years = [2020, 2021, 2022, 2023, 2024]
        result = permutation_test_changepoint(
            signal, years, n_permutations=50, seed=42
        )
        assert "changepoints" in result
        assert "n_permutations" in result
        assert result["n_permutations"] == 50

    def test_no_changepoints(self):
        signal = [5.0, 5.0, 5.0, 5.0, 5.0]
        years = [2020, 2021, 2022, 2023, 2024]
        result = permutation_test_changepoint(signal, years, n_permutations=20)
        assert result["changepoints"] == []

    def test_p_value_range(self):
        signal = [1.0, 2.0, 50.0, 48.0, 52.0]
        years = [2020, 2021, 2022, 2023, 2024]
        result = permutation_test_changepoint(signal, years, n_permutations=50)
        for cp in result["changepoints"]:
            assert 0.0 <= cp["p_value"] <= 1.0


class TestAggregateChangepoints:
    def test_consensus(self):
        evos = []
        # Create 5 programs, all with CP at 2024
        for i in range(5):
            evo = ProgramEvolution(
                fgos_code=f"09.0{i}.01",
                years=[2021, 2022, 2023, 2024],
                signals={"n_courses": [3.0, 3.0, 3.0, 30.0]},
                changepoints=[
                    ChangePoint(
                        index=3,
                        year_before=2023,
                        year_after=2024,
                        signal_name="n_courses",
                        cost_before=3.0,
                        cost_after=30.0,
                        magnitude=27.0,
                    )
                ],
            )
            evos.append(evo)

        agg = aggregate_changepoints(evos)
        assert agg["total_programs"] == 5
        assert agg["programs_with_changepoints"] == 5
        assert 2024 in agg["consensus_years"]

    def test_no_consensus(self):
        # Diverse changepoints, no consensus
        evos = [
            ProgramEvolution(
                fgos_code="A",
                years=[2021, 2022, 2023],
                signals={},
                changepoints=[
                    ChangePoint(index=1, year_before=2021, year_after=2022,
                                signal_name="x", cost_before=1, cost_after=5, magnitude=4)
                ],
            ),
            ProgramEvolution(
                fgos_code="B",
                years=[2022, 2023, 2024],
                signals={},
                changepoints=[
                    ChangePoint(index=1, year_before=2022, year_after=2023,
                                signal_name="x", cost_before=1, cost_after=5, magnitude=4)
                ],
            ),
        ]
        agg = aggregate_changepoints(evos)
        assert agg["total_programs"] == 2
        # With only 2 programs and threshold max(2, 30%*2)=2, a year needs 2 hits
        # Each year has only 1, so no consensus
        assert len(agg["consensus_years"]) == 0

    def test_empty(self):
        agg = aggregate_changepoints([])
        assert agg["total_programs"] == 0
        assert agg["total_changepoints"] == 0


class TestProgramEvolution:
    def test_to_dict(self):
        evo = ProgramEvolution(
            fgos_code="09.04.01",
            years=[2021, 2022, 2023],
            signals={"n_courses": [3.0, 5.0, 10.0]},
            changepoints=[
                ChangePoint(index=2, year_before=2022, year_after=2023,
                            signal_name="n_courses", cost_before=4.0,
                            cost_after=10.0, magnitude=6.0)
            ],
        )
        d = evo.to_dict()
        assert d["fgos_code"] == "09.04.01"
        assert d["n_years"] == 3
        assert len(d["changepoints"]) == 1
        assert d["changepoints"][0]["signal"] == "n_courses"

    def test_properties(self):
        evo = ProgramEvolution(
            fgos_code="test",
            years=[2021, 2022],
            signals={},
        )
        assert evo.n_years == 2
        assert not evo.has_changepoints
