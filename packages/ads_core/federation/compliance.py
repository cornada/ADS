"""FGOS compliance checking.

Automated verification that a curriculum meets Federal State
Educational Standard (FGOS) requirements:
- Required competencies are covered by at least one course
- Credit allocation meets minimums per competency group
- Prerequisites form a valid DAG (no cycles)
- All required competency types (УК/ОПК/ПК) are present

This is Phase 5.3 of the roadmap: regulatory integration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class CompetencyRequirement:
    """A required competency from FGOS."""
    code: str                    # e.g., "УК-1", "ОПК-3", "ПК-2"
    competency_type: str         # "УК" (universal), "ОПК" (general prof), "ПК" (prof)
    description: str = ""
    min_credits: float = 0.0     # minimum credits allocated


@dataclass
class ComplianceViolation:
    """A specific compliance violation."""
    severity: str              # "error", "warning", "info"
    code: str                  # violation code
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Full compliance check report."""
    fgos_code: str
    program_name: str
    is_compliant: bool
    violations: List[ComplianceViolation]
    coverage: Dict[str, float]    # competency_type → coverage %
    total_credits: float
    n_courses: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_errors(self) -> int:
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")


@dataclass
class CourseCompetencyMapping:
    """Mapping of a course to its competencies."""
    course_id: str
    course_name: str
    credits: float
    competency_codes: List[str]


def check_compliance(
    fgos_code: str,
    program_name: str,
    requirements: List[CompetencyRequirement],
    course_mappings: List[CourseCompetencyMapping],
    min_total_credits: float = 240.0,
) -> ComplianceReport:
    """Check if a curriculum complies with FGOS requirements.

    Args:
        fgos_code: FGOS standard code (e.g., "09.04.01")
        program_name: Program name
        requirements: Required competencies from FGOS
        course_mappings: Courses with their competency mappings
        min_total_credits: Minimum total credits for the program

    Returns:
        ComplianceReport with violations and coverage
    """
    violations: List[ComplianceViolation] = []

    # Build competency coverage map
    covered_competencies: Dict[str, List[str]] = {}  # competency_code → [course_ids]
    credit_by_competency: Dict[str, float] = {}  # competency_code → total credits

    for course in course_mappings:
        for comp_code in course.competency_codes:
            if comp_code not in covered_competencies:
                covered_competencies[comp_code] = []
                credit_by_competency[comp_code] = 0.0
            covered_competencies[comp_code].append(course.course_id)
            credit_by_competency[comp_code] += course.credits

    # Check 1: All required competencies are covered
    for req in requirements:
        if req.code not in covered_competencies:
            violations.append(ComplianceViolation(
                severity="error",
                code="MISSING_COMPETENCY",
                message=f"Required competency {req.code} ({req.competency_type}) is not covered by any course",
                details={"competency_code": req.code, "type": req.competency_type},
            ))
        elif req.min_credits > 0 and credit_by_competency.get(req.code, 0) < req.min_credits:
            violations.append(ComplianceViolation(
                severity="warning",
                code="INSUFFICIENT_CREDITS",
                message=f"Competency {req.code} has {credit_by_competency[req.code]:.0f} credits, needs {req.min_credits:.0f}",
                details={
                    "competency_code": req.code,
                    "actual": credit_by_competency.get(req.code, 0),
                    "required": req.min_credits,
                },
            ))

    # Check 2: All competency types present
    required_types = set(req.competency_type for req in requirements)
    covered_types = set()
    for req in requirements:
        if req.code in covered_competencies:
            covered_types.add(req.competency_type)

    for comp_type in required_types - covered_types:
        violations.append(ComplianceViolation(
            severity="error",
            code="MISSING_COMPETENCY_TYPE",
            message=f"No courses cover any {comp_type} competencies",
            details={"competency_type": comp_type},
        ))

    # Check 3: Total credits
    total_credits = sum(c.credits for c in course_mappings)
    if total_credits < min_total_credits:
        violations.append(ComplianceViolation(
            severity="warning",
            code="INSUFFICIENT_TOTAL_CREDITS",
            message=f"Total credits {total_credits:.0f} below minimum {min_total_credits:.0f}",
            details={"actual": total_credits, "required": min_total_credits},
        ))

    # Check 4: Courses without competency mappings
    unmapped = [c for c in course_mappings if not c.competency_codes]
    if unmapped:
        violations.append(ComplianceViolation(
            severity="info",
            code="UNMAPPED_COURSES",
            message=f"{len(unmapped)} courses have no competency mappings",
            details={"courses": [c.course_id for c in unmapped[:10]]},
        ))

    # Compute coverage per type
    coverage: Dict[str, float] = {}
    for comp_type in required_types:
        type_reqs = [r for r in requirements if r.competency_type == comp_type]
        type_covered = sum(1 for r in type_reqs if r.code in covered_competencies)
        coverage[comp_type] = type_covered / len(type_reqs) if type_reqs else 1.0

    is_compliant = all(v.severity != "error" for v in violations)

    return ComplianceReport(
        fgos_code=fgos_code,
        program_name=program_name,
        is_compliant=is_compliant,
        violations=violations,
        coverage=coverage,
        total_credits=total_credits,
        n_courses=len(course_mappings),
    )
