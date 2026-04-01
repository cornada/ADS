"""Career path navigator.

Given a learner's current competencies and a target job profile,
find the optimal curriculum path that closes the gap.

Uses shortest-path optimization in the course embedding space:
- Source: learner's current competency vector
- Target: job profile's required competency vector
- Edges: courses that move the learner closer to the target
- Costs: course credits (time), difficulty, prerequisites

This is Phase 5.4: learner-facing product.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CompetencyProfile:
    """A competency profile (learner or job target)."""
    profile_id: str
    competency_scores: Dict[str, float]  # competency_name → score [0, 1]
    label: str = ""

    def to_vector(self, competency_names: List[str]) -> np.ndarray:
        return np.array([self.competency_scores.get(name, 0.0) for name in competency_names])


@dataclass
class CourseEffect:
    """A course's effect on competencies."""
    course_id: str
    course_name: str
    credits: float
    competency_gains: Dict[str, float]  # competency_name → expected gain
    prerequisites: List[str] = field(default_factory=list)
    difficulty: float = 0.5  # [0, 1]

    def gain_vector(self, competency_names: List[str]) -> np.ndarray:
        return np.array([self.competency_gains.get(name, 0.0) for name in competency_names])


@dataclass
class PathStep:
    """A single step in a career path."""
    course: CourseEffect
    cumulative_credits: float
    competency_after: Dict[str, float]
    gap_reduction: float  # how much this step reduced the gap


@dataclass
class CareerPath:
    """An optimal curriculum path from current to target."""
    learner_id: str
    target_id: str
    steps: List[PathStep]
    total_credits: float
    initial_gap: float
    final_gap: float
    gap_closed_pct: float
    competency_names: List[str]


def compute_gap(current: np.ndarray, target: np.ndarray) -> float:
    """Compute competency gap (L2 distance, clamped to positive differences)."""
    deficit = np.maximum(target - current, 0)
    return float(np.linalg.norm(deficit))


def find_career_path(
    learner: CompetencyProfile,
    target: CompetencyProfile,
    courses: List[CourseEffect],
    competency_names: List[str],
    max_credits: float = 120.0,
    max_courses: int = 20,
    strategy: str = "greedy",
) -> CareerPath:
    """Find optimal curriculum path from learner's current state to target.

    Strategies:
    - "greedy": at each step, pick the course that maximally reduces the gap
    - "efficient": pick the course with best gap-reduction per credit
    - "breadth_first": prioritize courses covering the most competency gaps

    Args:
        learner: Current competency profile
        target: Target job profile
        courses: Available courses with their effects
        competency_names: Ordered list of competency dimensions
        max_credits: Maximum total credits
        max_courses: Maximum courses in path
        strategy: Selection strategy

    Returns:
        CareerPath with ordered list of courses
    """
    current = learner.to_vector(competency_names)
    target_vec = target.to_vector(competency_names)
    initial_gap = compute_gap(current, target_vec)

    steps: List[PathStep] = []
    total_credits = 0.0
    used_courses: set = set()
    completed: set = set()  # for prerequisite checking

    for _ in range(max_courses):
        if total_credits >= max_credits:
            break
        if compute_gap(current, target_vec) < 0.01:
            break  # gap closed

        # Find eligible courses (prerequisites met, not yet taken)
        eligible = [
            c for c in courses
            if c.course_id not in used_courses
            and all(p in completed for p in c.prerequisites)
            and total_credits + c.credits <= max_credits
        ]

        if not eligible:
            break

        # Score each course
        best_course = None
        best_score = -np.inf

        for course in eligible:
            gain = course.gain_vector(competency_names)
            new_state = np.minimum(current + gain, 1.0)
            new_gap = compute_gap(new_state, target_vec)
            gap_reduction = compute_gap(current, target_vec) - new_gap

            if strategy == "greedy":
                score = gap_reduction
            elif strategy == "efficient":
                score = gap_reduction / max(course.credits, 0.1)
            elif strategy == "breadth_first":
                deficit = np.maximum(target_vec - current, 0)
                # Count how many deficient dimensions this course addresses
                addresses = np.sum((gain > 0) & (deficit > 0))
                score = addresses + 0.1 * gap_reduction
            else:
                score = gap_reduction

            if score > best_score:
                best_score = score
                best_course = course

        if best_course is None or best_score <= 0:
            break

        # Apply course
        gain = best_course.gain_vector(competency_names)
        current = np.minimum(current + gain, 1.0)
        total_credits += best_course.credits
        used_courses.add(best_course.course_id)
        completed.add(best_course.course_id)

        comp_after = {name: float(current[i]) for i, name in enumerate(competency_names)}
        steps.append(PathStep(
            course=best_course,
            cumulative_credits=total_credits,
            competency_after=comp_after,
            gap_reduction=best_score if strategy == "greedy" else float(
                compute_gap(current - best_course.gain_vector(competency_names), target_vec)
                - compute_gap(current, target_vec)
            ),
        ))

    final_gap = compute_gap(current, target_vec)
    gap_closed = (1 - final_gap / initial_gap) * 100 if initial_gap > 0 else 100.0

    return CareerPath(
        learner_id=learner.profile_id,
        target_id=target.profile_id,
        steps=steps,
        total_credits=total_credits,
        initial_gap=initial_gap,
        final_gap=final_gap,
        gap_closed_pct=gap_closed,
        competency_names=competency_names,
    )
