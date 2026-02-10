"""Federated Pareto front computation.

Computes Pareto fronts across multiple institutions without sharing
raw data — only embedding-derived objective scores are exchanged.

Architecture:
- Each university computes local objective scores for its courses
- Scores are sent to a central coordinator (or peer-to-peer)
- Coordinator merges and computes the global Pareto front
- Results are broadcast back: each university sees where its courses
  rank globally and what gaps exist

Privacy model: embeddings are shared (not raw text), and only
aggregate statistics cross institutional boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class InstitutionSubmission:
    """Objective scores submitted by one institution."""
    institution_id: str
    course_ids: List[str]
    scores: np.ndarray       # (n_courses, n_objectives)
    objective_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_courses(self) -> int:
        return len(self.course_ids)


@dataclass
class FederatedParetoResult:
    """Result of federated Pareto computation."""
    pareto_indices: List[int]       # indices into merged array
    pareto_courses: List[str]       # course IDs on the front
    pareto_institutions: List[str]  # institution per Pareto course
    institution_counts: Dict[str, int]  # courses on front per institution
    total_courses: int
    n_objectives: int
    coverage_per_institution: Dict[str, float]  # % of front per institution


def merge_submissions(
    submissions: List[InstitutionSubmission],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Merge submissions from multiple institutions.

    Returns:
        merged_scores: (N_total, n_objectives) array
        merged_ids: course IDs
        merged_institutions: institution ID per course
    """
    if not submissions:
        return np.empty((0, 0)), [], []

    # Verify all submissions use the same objectives
    obj_names = submissions[0].objective_names
    for sub in submissions[1:]:
        if sub.objective_names != obj_names:
            raise ValueError(
                f"Objective mismatch: {sub.institution_id} has "
                f"{sub.objective_names}, expected {obj_names}"
            )

    all_scores = []
    all_ids = []
    all_institutions = []

    for sub in submissions:
        all_scores.append(sub.scores)
        all_ids.extend(sub.course_ids)
        all_institutions.extend([sub.institution_id] * sub.n_courses)

    merged = np.vstack(all_scores)
    return merged, all_ids, all_institutions


def compute_pareto_front(scores: np.ndarray) -> List[int]:
    """Compute Pareto front indices (all objectives maximized).

    A point is Pareto-optimal if no other point dominates it
    (i.e., is better in all objectives simultaneously).
    """
    n = scores.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i if j >= i in all objectives and j > i in at least one
            if np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                is_pareto[i] = False
                break

    return [i for i in range(n) if is_pareto[i]]


def federated_pareto(
    submissions: List[InstitutionSubmission],
) -> FederatedParetoResult:
    """Compute the global Pareto front from institutional submissions."""
    merged_scores, merged_ids, merged_institutions = merge_submissions(submissions)

    if len(merged_ids) == 0:
        return FederatedParetoResult(
            pareto_indices=[], pareto_courses=[], pareto_institutions=[],
            institution_counts={}, total_courses=0, n_objectives=0,
            coverage_per_institution={},
        )

    pareto_idx = compute_pareto_front(merged_scores)

    pareto_courses = [merged_ids[i] for i in pareto_idx]
    pareto_institutions = [merged_institutions[i] for i in pareto_idx]

    # Count per institution
    inst_counts: Dict[str, int] = {}
    for inst in pareto_institutions:
        inst_counts[inst] = inst_counts.get(inst, 0) + 1

    # Coverage: what fraction of the front comes from each institution
    n_front = len(pareto_idx)
    coverage = {inst: count / n_front for inst, count in inst_counts.items()} if n_front > 0 else {}

    return FederatedParetoResult(
        pareto_indices=pareto_idx,
        pareto_courses=pareto_courses,
        pareto_institutions=pareto_institutions,
        institution_counts=inst_counts,
        total_courses=len(merged_ids),
        n_objectives=merged_scores.shape[1],
        coverage_per_institution=coverage,
    )


def compute_institutional_gaps(
    result: FederatedParetoResult,
    submissions: List[InstitutionSubmission],
) -> Dict[str, Dict[str, Any]]:
    """Identify gaps for each institution relative to the global front.

    For each institution, computes:
    - Which objectives are under-represented on the front
    - Distance to nearest Pareto point for non-Pareto courses
    - Specific courses that are close to the front (improvement candidates)
    """
    merged_scores, merged_ids, merged_institutions = merge_submissions(submissions)
    if len(merged_ids) == 0:
        return {}

    pareto_set = set(result.pareto_indices)
    pareto_scores = merged_scores[result.pareto_indices]

    gaps = {}
    for sub in submissions:
        inst = sub.institution_id
        # Find this institution's non-Pareto courses
        inst_indices = [i for i, iid in enumerate(merged_institutions) if iid == inst]
        non_pareto = [i for i in inst_indices if i not in pareto_set]

        if not non_pareto or len(pareto_scores) == 0:
            gaps[inst] = {
                "n_on_front": result.institution_counts.get(inst, 0),
                "n_total": sub.n_courses,
                "near_front": [],
                "weakest_objectives": [],
            }
            continue

        # Distance of non-Pareto courses to nearest Pareto point
        distances = []
        for idx in non_pareto:
            dists = np.linalg.norm(pareto_scores - merged_scores[idx], axis=1)
            min_dist = float(np.min(dists))
            distances.append((idx, min_dist))

        distances.sort(key=lambda x: x[1])
        near_front = [
            {"course_id": merged_ids[idx], "distance": dist}
            for idx, dist in distances[:5]
        ]

        # Weakest objectives: where this institution's Pareto courses score lowest
        inst_pareto = [i for i in inst_indices if i in pareto_set]
        if inst_pareto:
            mean_scores = merged_scores[inst_pareto].mean(axis=0)
            global_mean = pareto_scores.mean(axis=0)
            obj_names = submissions[0].objective_names
            weakness = list(np.argsort(mean_scores - global_mean))
            weakest = [obj_names[i] for i in weakness[:2]]
        else:
            weakest = []

        gaps[inst] = {
            "n_on_front": result.institution_counts.get(inst, 0),
            "n_total": sub.n_courses,
            "near_front": near_front,
            "weakest_objectives": weakest,
        }

    return gaps
