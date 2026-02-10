"""Tests for Phase 5 federation, compliance, and career path modules."""
import numpy as np
import pytest

from ads_core.federation.federated_pareto import (
    InstitutionSubmission,
    FederatedParetoResult,
    merge_submissions,
    compute_pareto_front,
    federated_pareto,
    compute_institutional_gaps,
)
from ads_core.federation.compliance import (
    CompetencyRequirement,
    CourseCompetencyMapping,
    ComplianceReport,
    check_compliance,
)
from ads_core.federation.career_path import (
    CompetencyProfile,
    CourseEffect,
    CareerPath,
    compute_gap,
    find_career_path,
)


# ============================================================================
# Federated Pareto
# ============================================================================

class TestFederatedPareto:
    @pytest.fixture
    def two_universities(self):
        """Two institutions with 3 objectives."""
        sub_a = InstitutionSubmission(
            institution_id="MISIS",
            course_ids=["m1", "m2", "m3"],
            scores=np.array([
                [0.9, 0.5, 0.3],  # m1: strong market
                [0.3, 0.9, 0.5],  # m2: strong mission
                [0.5, 0.5, 0.9],  # m3: strong coverage
            ]),
            objective_names=["market", "mission", "coverage"],
        )
        sub_b = InstitutionSubmission(
            institution_id="KTH",
            course_ids=["k1", "k2"],
            scores=np.array([
                [0.8, 0.8, 0.4],  # k1: balanced
                [0.4, 0.4, 0.4],  # k2: dominated
            ]),
            objective_names=["market", "mission", "coverage"],
        )
        return [sub_a, sub_b]

    def test_merge_submissions(self, two_universities):
        scores, ids, institutions = merge_submissions(two_universities)
        assert scores.shape == (5, 3)
        assert len(ids) == 5
        assert institutions.count("MISIS") == 3
        assert institutions.count("KTH") == 2

    def test_pareto_front_simple(self):
        scores = np.array([
            [1, 0],  # Pareto
            [0, 1],  # Pareto
            [0.5, 0.5],  # dominated by neither → Pareto
            [0.3, 0.3],  # dominated by (0.5, 0.5)
        ])
        front = compute_pareto_front(scores)
        assert 0 in front
        assert 1 in front
        assert 2 in front
        assert 3 not in front

    def test_federated_pareto(self, two_universities):
        result = federated_pareto(two_universities)
        assert result.total_courses == 5
        assert len(result.pareto_courses) > 0
        assert len(result.pareto_courses) <= 5
        # k2 should be dominated
        assert "k2" not in result.pareto_courses

    def test_federated_pareto_empty(self):
        result = federated_pareto([])
        assert result.total_courses == 0

    def test_institutional_gaps(self, two_universities):
        result = federated_pareto(two_universities)
        gaps = compute_institutional_gaps(result, two_universities)
        assert "MISIS" in gaps
        assert "KTH" in gaps
        assert gaps["MISIS"]["n_total"] == 3
        assert gaps["KTH"]["n_total"] == 2

    def test_objective_mismatch_raises(self):
        sub_a = InstitutionSubmission(
            institution_id="A", course_ids=["a1"],
            scores=np.array([[0.5, 0.5]]),
            objective_names=["x", "y"],
        )
        sub_b = InstitutionSubmission(
            institution_id="B", course_ids=["b1"],
            scores=np.array([[0.5, 0.5]]),
            objective_names=["x", "z"],  # different!
        )
        with pytest.raises(ValueError, match="Objective mismatch"):
            federated_pareto([sub_a, sub_b])


# ============================================================================
# FGOS Compliance
# ============================================================================

class TestCompliance:
    @pytest.fixture
    def fgos_requirements(self):
        return [
            CompetencyRequirement("УК-1", "УК", "Critical thinking"),
            CompetencyRequirement("УК-2", "УК", "Project management"),
            CompetencyRequirement("ОПК-1", "ОПК", "Scientific research", min_credits=6),
            CompetencyRequirement("ПК-1", "ПК", "Programming", min_credits=12),
            CompetencyRequirement("ПК-2", "ПК", "System design"),
        ]

    @pytest.fixture
    def compliant_courses(self):
        return [
            CourseCompetencyMapping("c1", "Critical Thinking", 3, ["УК-1"]),
            CourseCompetencyMapping("c2", "Project Lab", 3, ["УК-2"]),
            CourseCompetencyMapping("c3", "Research Methods", 6, ["ОПК-1"]),
            CourseCompetencyMapping("c4", "Algorithms", 6, ["ПК-1"]),
            CourseCompetencyMapping("c5", "Programming", 6, ["ПК-1"]),
            CourseCompetencyMapping("c6", "System Design", 6, ["ПК-2"]),
            CourseCompetencyMapping("c7", "Elective", 3, []),
        ]

    def test_compliant_program(self, fgos_requirements, compliant_courses):
        report = check_compliance(
            "09.04.01", "Информатика", fgos_requirements, compliant_courses,
            min_total_credits=30,
        )
        assert report.is_compliant
        assert report.n_errors == 0
        assert report.n_courses == 7
        assert report.total_credits == 33

    def test_missing_competency(self, fgos_requirements):
        # No course covers ПК-2
        courses = [
            CourseCompetencyMapping("c1", "CT", 3, ["УК-1"]),
            CourseCompetencyMapping("c2", "PM", 3, ["УК-2"]),
            CourseCompetencyMapping("c3", "RM", 6, ["ОПК-1"]),
            CourseCompetencyMapping("c4", "Algo", 12, ["ПК-1"]),
        ]
        report = check_compliance(
            "09.04.01", "Test", fgos_requirements, courses,
            min_total_credits=20,
        )
        assert not report.is_compliant
        assert report.n_errors >= 1
        assert any(v.code == "MISSING_COMPETENCY" for v in report.violations)

    def test_insufficient_credits(self, fgos_requirements):
        courses = [
            CourseCompetencyMapping("c1", "CT", 3, ["УК-1"]),
            CourseCompetencyMapping("c2", "PM", 3, ["УК-2"]),
            CourseCompetencyMapping("c3", "RM", 3, ["ОПК-1"]),  # Only 3, needs 6
            CourseCompetencyMapping("c4", "Algo", 6, ["ПК-1"]),  # Only 6, needs 12
            CourseCompetencyMapping("c5", "SD", 3, ["ПК-2"]),
        ]
        report = check_compliance("09.04.01", "Test", fgos_requirements, courses)
        warnings = [v for v in report.violations if v.code == "INSUFFICIENT_CREDITS"]
        assert len(warnings) >= 1

    def test_coverage_per_type(self, fgos_requirements, compliant_courses):
        report = check_compliance(
            "09.04.01", "Test", fgos_requirements, compliant_courses,
            min_total_credits=30,
        )
        assert report.coverage["УК"] == 1.0
        assert report.coverage["ОПК"] == 1.0
        assert report.coverage["ПК"] == 1.0

    def test_unmapped_courses_info(self, fgos_requirements, compliant_courses):
        report = check_compliance(
            "09.04.01", "Test", fgos_requirements, compliant_courses,
            min_total_credits=30,
        )
        # c7 has no competencies
        info = [v for v in report.violations if v.code == "UNMAPPED_COURSES"]
        assert len(info) == 1


# ============================================================================
# Career Path
# ============================================================================

class TestCareerPath:
    @pytest.fixture
    def setup(self):
        names = ["programming", "algorithms", "systems"]
        learner = CompetencyProfile("student_1", {"programming": 0.3, "algorithms": 0.2, "systems": 0.1})
        target = CompetencyProfile("swe_job", {"programming": 0.8, "algorithms": 0.7, "systems": 0.6})
        courses = [
            CourseEffect("cs101", "Intro CS", 3, {"programming": 0.2}),
            CourseEffect("cs201", "Data Structures", 3, {"algorithms": 0.3}, prerequisites=["cs101"]),
            CourseEffect("cs301", "Systems", 3, {"systems": 0.3}),
            CourseEffect("cs401", "Advanced Programming", 3, {"programming": 0.3}, prerequisites=["cs201"]),
            CourseEffect("cs501", "OS", 3, {"systems": 0.2, "programming": 0.1}, prerequisites=["cs301"]),
        ]
        return names, learner, target, courses

    def test_compute_gap(self):
        current = np.array([0.3, 0.2])
        target = np.array([0.8, 0.7])
        gap = compute_gap(current, target)
        assert gap > 0
        # If current >= target, gap should be 0
        assert compute_gap(np.array([1.0, 1.0]), target) == 0.0

    def test_find_path_greedy(self, setup):
        names, learner, target, courses = setup
        path = find_career_path(learner, target, courses, names, strategy="greedy")
        assert len(path.steps) > 0
        assert path.total_credits > 0
        assert path.gap_closed_pct > 0

    def test_find_path_efficient(self, setup):
        names, learner, target, courses = setup
        path = find_career_path(learner, target, courses, names, strategy="efficient")
        assert len(path.steps) > 0

    def test_find_path_breadth_first(self, setup):
        names, learner, target, courses = setup
        path = find_career_path(learner, target, courses, names, strategy="breadth_first")
        assert len(path.steps) > 0

    def test_prerequisites_respected(self, setup):
        names, learner, target, courses = setup
        path = find_career_path(learner, target, courses, names)
        taken_ids = set()
        for step in path.steps:
            for prereq in step.course.prerequisites:
                assert prereq in taken_ids, f"{step.course.course_id} taken before prereq {prereq}"
            taken_ids.add(step.course.course_id)

    def test_max_credits_respected(self, setup):
        names, learner, target, courses = setup
        path = find_career_path(learner, target, courses, names, max_credits=6)
        assert path.total_credits <= 6

    def test_already_qualified(self):
        names = ["a", "b"]
        learner = CompetencyProfile("l1", {"a": 0.9, "b": 0.9})
        target = CompetencyProfile("t1", {"a": 0.5, "b": 0.5})
        courses = [CourseEffect("c1", "C1", 3, {"a": 0.1})]
        path = find_career_path(learner, target, courses, names)
        assert path.gap_closed_pct == 100.0
        assert len(path.steps) == 0
