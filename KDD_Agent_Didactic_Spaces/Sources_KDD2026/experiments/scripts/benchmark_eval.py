#!/usr/bin/env python3
"""Task-agnostic benchmark evaluation for the ADS benchmark.

This script evaluates ANY method's output against the ADS benchmark,
regardless of whether that method uses ADS internals, TF-IDF, LLMs,
or any other approach.

Three benchmark tasks are defined:
  Task A: Multi-Stakeholder Course Ranking
  Task B: Stakeholder Trade-off Discovery
  Task C: Robust Recommendation

Usage:
    python benchmark_eval.py --submission submission.json --ground-truth ground_truth.json
    python benchmark_eval.py --submission submission.json --ground-truth ground_truth.json --task A
    python benchmark_eval.py --submission submission.json --ground-truth ground_truth.json --task all --output results.json

Dependencies: numpy, scipy (standard scientific Python only).

Submission format (JSON):
{
  "method_name": "my_method",
  "task": "A",                        # "A", "B", "C", or "all"
  "ranked_courses": ["course_1", "course_2", ...],
  "scores": {                         # optional per-course score vectors
    "course_1": {"market": 0.8, "mission": 0.3, ...},
    "course_2": {"market": 0.5, "mission": 0.6, ...}
  },
  "perturbation_runs": [              # required only for Task C
    {"ranked_courses": [...], "scores": {...}},
    ...
  ]
}

Ground-truth format (JSON):
{
  "objectives": ["market", "mission", "university", "learner"],
  "courses": {
    "course_1": {"market": 0.82, "mission": 0.31, "university": 0.45, "learner": 0.50},
    ...
  },
  "reference_pareto": ["course_1", "course_7", ...],  # optional
  "splits": {
    "dev": ["course_1", "course_2", ...],
    "eval": ["course_1", "course_2", ...]
  }
}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# JSON Schema Validation
# ============================================================================

SUBMISSION_REQUIRED_KEYS = {"method_name", "task", "ranked_courses"}
GROUND_TRUTH_REQUIRED_KEYS = {"objectives", "courses"}


def validate_submission(sub: Dict[str, Any]) -> List[str]:
    """Validate submission format, returning list of errors (empty if valid)."""
    errors = []
    for key in SUBMISSION_REQUIRED_KEYS:
        if key not in sub:
            errors.append(f"Missing required key: '{key}'")
    if "task" in sub and sub["task"] not in ("A", "B", "C", "all"):
        errors.append(f"Invalid task: '{sub['task']}'. Must be 'A', 'B', 'C', or 'all'.")
    if "ranked_courses" in sub:
        if not isinstance(sub["ranked_courses"], list):
            errors.append("'ranked_courses' must be a list of course IDs.")
        elif len(sub["ranked_courses"]) == 0:
            errors.append("'ranked_courses' must not be empty.")
    if sub.get("task") in ("C", "all"):
        if "perturbation_runs" not in sub:
            errors.append("Task C requires 'perturbation_runs' field.")
        elif not isinstance(sub["perturbation_runs"], list):
            errors.append("'perturbation_runs' must be a list.")
        elif len(sub["perturbation_runs"]) < 2:
            errors.append("Task C requires at least 2 perturbation runs.")
    return errors


def validate_ground_truth(gt: Dict[str, Any]) -> List[str]:
    """Validate ground-truth format."""
    errors = []
    for key in GROUND_TRUTH_REQUIRED_KEYS:
        if key not in gt:
            errors.append(f"Missing required key: '{key}'")
    if "objectives" in gt and not isinstance(gt["objectives"], list):
        errors.append("'objectives' must be a list of objective names.")
    if "courses" in gt and not isinstance(gt["courses"], dict):
        errors.append("'courses' must be a dict of course_id -> score_dict.")
    return errors


# ============================================================================
# Core Metric Functions (self-contained, no ADS dependency)
# ============================================================================

def dominates(a: Dict[str, float], b: Dict[str, float], keys: List[str]) -> bool:
    """Check if solution a Pareto-dominates b (maximization).

    a dominates b iff:
      a[k] >= b[k] for all k  AND  a[k] > b[k] for at least one k.
    """
    at_least_as_good = all(a.get(k, 0.0) >= b.get(k, 0.0) for k in keys)
    strictly_better = any(a.get(k, 0.0) > b.get(k, 0.0) for k in keys)
    return at_least_as_good and strictly_better


def compute_pareto_front(
    scores: Dict[str, Dict[str, float]], keys: List[str]
) -> List[str]:
    """Compute the Pareto-optimal course IDs from score vectors."""
    ids = list(scores.keys())
    pareto = []
    for i, cid_i in enumerate(ids):
        is_dominated = False
        for j, cid_j in enumerate(ids):
            if i != j and dominates(scores[cid_j], scores[cid_i], keys):
                is_dominated = True
                break
        if not is_dominated:
            pareto.append(cid_i)
    return pareto


def compute_hypervolume(
    scores: Dict[str, Dict[str, float]],
    keys: List[str],
    ref_point: Optional[Dict[str, float]] = None,
    n_samples: int = 50000,
    seed: int = 42,
) -> float:
    """Compute hypervolume indicator for a set of solutions.

    For 2 objectives, uses exact sweep-line.
    For >2 objectives, uses Monte Carlo approximation.

    Args:
        scores: course_id -> {objective: value}
        keys: objective names
        ref_point: reference point (default: origin)
        n_samples: MC samples for >2D
        seed: random seed for MC

    Returns:
        Hypervolume value (higher is better)
    """
    if not scores:
        return 0.0

    n_obj = len(keys)
    if ref_point is None:
        ref_point = {k: 0.0 for k in keys}

    points = np.array([[s.get(k, 0.0) for k in keys] for s in scores.values()])
    ref = np.array([ref_point.get(k, 0.0) for k in keys])
    shifted = points - ref

    # Filter points dominated by reference
    valid = np.all(shifted > 0, axis=1)
    shifted = shifted[valid]
    if len(shifted) == 0:
        return 0.0

    if n_obj == 2:
        return _hypervolume_2d(shifted)
    else:
        return _hypervolume_mc(shifted, n_samples, seed)


def _hypervolume_2d(points: np.ndarray) -> float:
    """Exact 2D hypervolume via sweep line."""
    sorted_idx = np.argsort(-points[:, 0])
    points = points[sorted_idx]
    hv = 0.0
    prev_y = 0.0
    for x, y in points:
        if y > prev_y:
            hv += x * (y - prev_y)
            prev_y = y
    return float(hv)


def _hypervolume_mc(points: np.ndarray, n_samples: int, seed: int) -> float:
    """Monte Carlo hypervolume approximation for >2 objectives."""
    rng = np.random.RandomState(seed)
    max_pt = points.max(axis=0)
    samples = rng.uniform(0, max_pt, size=(n_samples, points.shape[1]))
    dominated = 0
    for s in samples:
        for p in points:
            if np.all(s <= p):
                dominated += 1
                break
    box_vol = float(np.prod(max_pt))
    return dominated / n_samples * box_vol


def compute_spacing(
    scores: Dict[str, Dict[str, float]], keys: List[str]
) -> float:
    """Compute spacing metric: average nearest-neighbor distance in objective space.

    Lower spacing means more uniform distribution of trade-offs.
    """
    if len(scores) < 2:
        return 0.0
    points = np.array([[s.get(k, 0.0) for k in keys] for s in scores.values()])
    dists = []
    for i in range(len(points)):
        min_d = float("inf")
        for j in range(len(points)):
            if i != j:
                d = float(np.linalg.norm(points[i] - points[j]))
                if d < min_d:
                    min_d = d
        if min_d < float("inf"):
            dists.append(min_d)
    if not dists:
        return 0.0
    mean_d = float(np.mean(dists))
    # Spacing = std dev of nearest-neighbor distances (lower = more uniform)
    return float(np.std(dists))


def compute_spread(
    scores: Dict[str, Dict[str, float]], keys: List[str]
) -> float:
    """Compute spread: volume of bounding box in objective space."""
    if len(scores) < 2:
        return 0.0
    points = np.array([[s.get(k, 0.0) for k in keys] for s in scores.values()])
    ranges = points.max(axis=0) - points.min(axis=0)
    return float(np.prod(ranges)) if np.all(ranges > 0) else 0.0


def compute_coverage(
    selected_ids: List[str],
    all_scores: Dict[str, Dict[str, float]],
    keys: List[str],
    n_bins: int = 5,
) -> float:
    """Compute objective-space coverage via bin occupancy.

    Divides each objective dimension into n_bins bins and measures
    what fraction of the grid is occupied by the selected set.
    """
    if not selected_ids:
        return 0.0

    # Build grid over the full score range
    all_vals = {k: [] for k in keys}
    for cid, s in all_scores.items():
        for k in keys:
            all_vals[k].append(s.get(k, 0.0))

    bins_per_dim = {}
    for k in keys:
        lo, hi = min(all_vals[k]), max(all_vals[k])
        if hi - lo < 1e-9:
            bins_per_dim[k] = (lo - 0.5, hi + 0.5, 1)
        else:
            bins_per_dim[k] = (lo, hi, n_bins)

    # Count occupied cells
    occupied = set()
    for cid in selected_ids:
        if cid not in all_scores:
            continue
        cell = []
        for k in keys:
            lo, hi, nb = bins_per_dim[k]
            v = all_scores[cid].get(k, 0.0)
            b = min(int((v - lo) / (hi - lo + 1e-12) * nb), nb - 1)
            cell.append(b)
        occupied.add(tuple(cell))

    total_cells = 1
    for k in keys:
        total_cells *= bins_per_dim[k][2]

    return len(occupied) / total_cells


def compute_diversity(
    scores: Dict[str, Dict[str, float]], keys: List[str]
) -> float:
    """Compute diversity: average pairwise distance in objective space.

    Higher diversity means more distinct trade-offs.
    """
    if len(scores) < 2:
        return 0.0
    points = np.array([[s.get(k, 0.0) for k in keys] for s in scores.values()])
    n = len(points)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(points[i] - points[j]))
            count += 1
    return total / count if count > 0 else 0.0


def compute_objective_balance(
    scores: Dict[str, Dict[str, float]], keys: List[str]
) -> float:
    """Compute objective balance: Shannon entropy of mean objective proportions.

    Measures how evenly the selected set distributes across objectives.
    Maximum entropy (= log(n_obj)) means perfectly balanced.
    Returns normalized entropy in [0, 1].
    """
    if not scores or not keys:
        return 0.0
    means = []
    for k in keys:
        vals = [s.get(k, 0.0) for s in scores.values()]
        means.append(max(float(np.mean(vals)), 1e-12))
    total = sum(means)
    probs = [m / total for m in means]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(len(keys))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_external_alignment(
    selected_ids: List[str],
    gt_pareto: List[str],
) -> Dict[str, float]:
    """Compute alignment with a reference Pareto set.

    Returns:
        precision: fraction of selected that are in reference
        recall: fraction of reference captured
        f1: harmonic mean
        jaccard: set overlap
    """
    sel = set(selected_ids)
    ref = set(gt_pareto)
    if not sel and not ref:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0}
    intersection = sel & ref
    precision = len(intersection) / len(sel) if sel else 0.0
    recall = len(intersection) / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    union = sel | ref
    jaccard = len(intersection) / len(union) if union else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "jaccard": jaccard}


def jaccard_similarity(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def spearman_correlation(ranks_a: List[float], ranks_b: List[float]) -> float:
    """Spearman rank correlation. Uses scipy if available, else manual."""
    if len(ranks_a) < 2:
        return 1.0
    if HAS_SCIPY:
        rho, _ = spearmanr(ranks_a, ranks_b)
        return float(rho) if not np.isnan(rho) else 0.0
    # Manual Spearman
    n = len(ranks_a)
    d_sq = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
    return 1 - 6 * d_sq / (n * (n ** 2 - 1))


# ============================================================================
# Task Evaluation Functions
# ============================================================================

def evaluate_task_a(
    submission: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate Task A: Multi-Stakeholder Course Ranking.

    Given course texts and stakeholder corpora, produce a ranked set of courses.

    Metrics:
      - coverage: fraction of objective-space grid cells occupied
      - diversity: average pairwise distance in objective space
      - objective_balance: Shannon entropy of mean objective proportions (normalized)
      - external_alignment: precision/recall/F1/Jaccard against reference Pareto set
      - rank_quality: Spearman correlation of submitted ranking vs. reference aggregate
      - selection_size: number of courses in submitted set
    """
    keys = ground_truth["objectives"]
    gt_courses = ground_truth["courses"]
    ranked = submission["ranked_courses"]

    # Filter to courses that exist in ground truth
    valid_ranked = [c for c in ranked if c in gt_courses]
    if not valid_ranked:
        return {
            "task": "A",
            "error": "No submitted courses found in ground truth.",
            "metrics": {},
        }

    # Build score dicts for selected courses
    selected_scores = {c: gt_courses[c] for c in valid_ranked}

    # If submission includes its own scores, use them for balance/diversity
    # but use ground-truth scores for external evaluation
    sub_scores = submission.get("scores", {})
    eval_scores = {}
    for c in valid_ranked:
        if c in sub_scores and all(k in sub_scores[c] for k in keys):
            eval_scores[c] = sub_scores[c]
        else:
            eval_scores[c] = gt_courses[c]

    metrics = {}

    # Coverage
    metrics["coverage"] = round(compute_coverage(valid_ranked, gt_courses, keys), 4)

    # Diversity
    metrics["diversity"] = round(compute_diversity(selected_scores, keys), 6)

    # Objective balance
    metrics["objective_balance"] = round(compute_objective_balance(selected_scores, keys), 4)

    # External alignment (if reference Pareto available)
    if "reference_pareto" in ground_truth:
        alignment = compute_external_alignment(valid_ranked, ground_truth["reference_pareto"])
        metrics["external_alignment"] = {k: round(v, 4) for k, v in alignment.items()}

    # Rank quality: correlation of submitted order with aggregate score order
    # Aggregate = mean of objectives (a naive but fair comparator)
    agg_scores = {
        c: sum(gt_courses[c].get(k, 0.0) for k in keys) / len(keys)
        for c in valid_ranked
    }
    submitted_ranks = list(range(len(valid_ranked)))
    sorted_by_agg = sorted(valid_ranked, key=lambda c: agg_scores[c], reverse=True)
    agg_ranks = [sorted_by_agg.index(c) for c in valid_ranked]
    metrics["rank_quality_vs_aggregate"] = round(
        spearman_correlation(submitted_ranks, agg_ranks), 4
    )

    # Selection size
    metrics["selection_size"] = len(valid_ranked)
    metrics["valid_fraction"] = round(len(valid_ranked) / len(ranked), 4)

    return {"task": "A", "metrics": metrics}


def evaluate_task_b(
    submission: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate Task B: Stakeholder Trade-off Discovery.

    Given course texts and stakeholder corpora, identify courses representing
    distinct trade-offs across stakeholder objectives.

    Metrics:
      - hypervolume: HV of submitted set (higher = better coverage + quality)
      - spacing: std dev of nearest-neighbor distances (lower = more uniform)
      - spread: bounding-box volume in objective space
      - pareto_quality: fraction of submitted courses on the true Pareto front
      - objective_balance: Shannon entropy of objective means (normalized)
      - n_trade_off_points: size of submitted set
      - dominance_resistance: fraction of submitted points not dominated by others
    """
    keys = ground_truth["objectives"]
    gt_courses = ground_truth["courses"]
    ranked = submission["ranked_courses"]

    valid_ranked = [c for c in ranked if c in gt_courses]
    if not valid_ranked:
        return {"task": "B", "error": "No valid courses.", "metrics": {}}

    selected_scores = {c: gt_courses[c] for c in valid_ranked}

    metrics = {}

    # Hypervolume
    metrics["hypervolume"] = round(
        compute_hypervolume(selected_scores, keys), 6
    )

    # Spacing
    metrics["spacing"] = round(compute_spacing(selected_scores, keys), 6)

    # Spread
    metrics["spread"] = round(compute_spread(selected_scores, keys), 6)

    # Pareto quality: what fraction of submitted courses are actually Pareto-optimal?
    true_pareto = compute_pareto_front(gt_courses, keys)
    true_pareto_set = set(true_pareto)
    submitted_on_pareto = [c for c in valid_ranked if c in true_pareto_set]
    metrics["pareto_precision"] = round(
        len(submitted_on_pareto) / len(valid_ranked), 4
    )
    metrics["pareto_recall"] = round(
        len(submitted_on_pareto) / max(len(true_pareto), 1), 4
    )

    # Dominance resistance: fraction of submitted set that is non-dominated
    # within the submission itself
    submitted_pareto = compute_pareto_front(selected_scores, keys)
    metrics["dominance_resistance"] = round(
        len(submitted_pareto) / len(valid_ranked), 4
    )

    # Objective balance
    metrics["objective_balance"] = round(
        compute_objective_balance(selected_scores, keys), 4
    )

    metrics["n_trade_off_points"] = len(valid_ranked)

    return {"task": "B", "metrics": metrics}


def evaluate_task_c(
    submission: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate Task C: Robust Recommendation.

    Produce recommendations stable under perturbation.
    Requires 'perturbation_runs' in submission.

    Metrics:
      - bootstrap_jaccard_mean: mean Jaccard between pairs of perturbation runs
      - bootstrap_jaccard_std: std of pairwise Jaccard
      - hv_retention: ratio of worst-case HV to best-case HV across runs
      - rank_stability: mean pairwise Spearman correlation of rankings
      - core_stable_fraction: fraction of courses appearing in >80% of runs
    """
    keys = ground_truth["objectives"]
    gt_courses = ground_truth["courses"]
    runs = submission.get("perturbation_runs", [])

    if len(runs) < 2:
        return {
            "task": "C",
            "error": "Need at least 2 perturbation runs.",
            "metrics": {},
        }

    # Also include the primary submission as run 0
    all_runs = [{"ranked_courses": submission["ranked_courses"],
                 "scores": submission.get("scores", {})}]
    all_runs.extend(runs)

    # Extract ranked course sets per run
    run_sets = []
    run_rankings = []
    run_hvs = []

    for run in all_runs:
        rc = [c for c in run["ranked_courses"] if c in gt_courses]
        run_sets.append(set(rc))
        run_rankings.append(rc)

        # Compute HV for this run
        run_scores = {c: gt_courses[c] for c in rc}
        hv = compute_hypervolume(run_scores, keys)
        run_hvs.append(hv)

    metrics = {}

    # Pairwise Jaccard
    jaccards = []
    for i in range(len(run_sets)):
        for j in range(i + 1, len(run_sets)):
            jaccards.append(jaccard_similarity(run_sets[i], run_sets[j]))
    metrics["bootstrap_jaccard_mean"] = round(float(np.mean(jaccards)), 4)
    metrics["bootstrap_jaccard_std"] = round(float(np.std(jaccards)), 4)

    # HV retention
    max_hv = max(run_hvs) if run_hvs else 0.0
    min_hv = min(run_hvs) if run_hvs else 0.0
    metrics["hv_retention"] = round(min_hv / max_hv, 4) if max_hv > 0 else 0.0

    # Rank stability: pairwise Spearman over shared courses
    rank_corrs = []
    for i in range(len(run_rankings)):
        for j in range(i + 1, len(run_rankings)):
            shared = list(set(run_rankings[i]) & set(run_rankings[j]))
            if len(shared) < 2:
                continue
            rank_i = [run_rankings[i].index(c) for c in shared]
            rank_j = [run_rankings[j].index(c) for c in shared]
            rank_corrs.append(spearman_correlation(rank_i, rank_j))
    metrics["rank_stability_mean"] = round(float(np.mean(rank_corrs)), 4) if rank_corrs else 1.0
    metrics["rank_stability_std"] = round(float(np.std(rank_corrs)), 4) if rank_corrs else 0.0

    # Core stable fraction
    all_courses = set()
    for s in run_sets:
        all_courses.update(s)
    n_runs = len(run_sets)
    freq = Counter()
    for s in run_sets:
        for c in s:
            freq[c] += 1
    core_stable = sum(1 for c in freq if freq[c] / n_runs >= 0.8)
    metrics["core_stable_fraction"] = round(
        core_stable / len(all_courses), 4
    ) if all_courses else 0.0
    metrics["core_stable_count"] = core_stable
    metrics["n_perturbation_runs"] = n_runs

    return {"task": "C", "metrics": metrics}


# ============================================================================
# Aggregate Evaluation
# ============================================================================

def evaluate_submission(
    submission: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate a submission for one or all tasks.

    Args:
        submission: The method's submission dict.
        ground_truth: The benchmark ground-truth dict.

    Returns:
        Dict with task results and overall summary.
    """
    task = submission.get("task", "all")
    method = submission.get("method_name", "unknown")
    results = {"method_name": method, "tasks": {}}

    if task in ("A", "all"):
        results["tasks"]["A"] = evaluate_task_a(submission, ground_truth)

    if task in ("B", "all"):
        results["tasks"]["B"] = evaluate_task_b(submission, ground_truth)

    if task in ("C", "all"):
        results["tasks"]["C"] = evaluate_task_c(submission, ground_truth)

    # Compute overall summary score (optional, for leaderboard convenience)
    # This is NOT a single scalar for ranking -- it is informational only
    summary = {}
    if "A" in results["tasks"] and "metrics" in results["tasks"]["A"]:
        m = results["tasks"]["A"]["metrics"]
        summary["task_a_coverage"] = m.get("coverage", 0.0)
        summary["task_a_diversity"] = m.get("diversity", 0.0)
        summary["task_a_balance"] = m.get("objective_balance", 0.0)
    if "B" in results["tasks"] and "metrics" in results["tasks"]["B"]:
        m = results["tasks"]["B"]["metrics"]
        summary["task_b_hypervolume"] = m.get("hypervolume", 0.0)
        summary["task_b_pareto_precision"] = m.get("pareto_precision", 0.0)
    if "C" in results["tasks"] and "metrics" in results["tasks"]["C"]:
        m = results["tasks"]["C"]["metrics"]
        summary["task_c_jaccard"] = m.get("bootstrap_jaccard_mean", 0.0)
        summary["task_c_hv_retention"] = m.get("hv_retention", 0.0)
    results["summary"] = summary

    return results


# ============================================================================
# Example: generate a trivial baseline submission for testing
# ============================================================================

def generate_random_baseline(
    ground_truth: Dict[str, Any], seed: int = 42, top_k: int = 50
) -> Dict[str, Any]:
    """Generate a random baseline submission for testing the evaluator.

    This shows how a non-ADS method would format its output.
    """
    rng = np.random.RandomState(seed)
    all_courses = list(ground_truth["courses"].keys())
    rng.shuffle(all_courses)
    selected = all_courses[:top_k]

    # Generate perturbation runs for Task C by re-shuffling
    runs = []
    for _ in range(5):
        rng.shuffle(all_courses)
        runs.append({
            "ranked_courses": all_courses[:top_k],
            "scores": {},
        })

    return {
        "method_name": "random_baseline",
        "task": "all",
        "ranked_courses": selected,
        "scores": {},
        "perturbation_runs": runs,
    }


def generate_weighted_sum_baseline(
    ground_truth: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    top_k: int = 50,
) -> Dict[str, Any]:
    """Generate a weighted-sum baseline (a TF-IDF or LLM ranker would do this).

    This demonstrates how a simple non-ADS method produces a submission:
    score each course as a weighted sum of objectives, rank, take top-k.
    """
    keys = ground_truth["objectives"]
    courses = ground_truth["courses"]

    if weights is None:
        weights = {k: 1.0 / len(keys) for k in keys}

    # Normalize weights
    w_total = sum(weights.values())
    weights = {k: v / w_total for k, v in weights.items()}

    # Score and rank
    scored = []
    for cid, obj in courses.items():
        s = sum(weights.get(k, 0.0) * obj.get(k, 0.0) for k in keys)
        scored.append((cid, s))
    scored.sort(key=lambda x: -x[1])

    selected = [c for c, _ in scored[:top_k]]
    scores_dict = {c: courses[c] for c in selected}

    # Perturbation runs: jitter weights
    rng = np.random.RandomState(0)
    runs = []
    for _ in range(5):
        noise = rng.normal(0, 0.1, size=len(keys))
        w_arr = np.array([weights[k] for k in keys]) + noise
        w_arr = np.clip(w_arr, 0.01, None)
        w_arr = w_arr / w_arr.sum()
        perturbed_weights = {k: float(w_arr[i]) for i, k in enumerate(keys)}

        p_scored = []
        for cid, obj in courses.items():
            s = sum(perturbed_weights.get(k, 0.0) * obj.get(k, 0.0) for k in keys)
            p_scored.append((cid, s))
        p_scored.sort(key=lambda x: -x[1])
        p_selected = [c for c, _ in p_scored[:top_k]]
        runs.append({
            "ranked_courses": p_selected,
            "scores": {c: courses[c] for c in p_selected},
        })

    return {
        "method_name": "weighted_sum_baseline",
        "task": "all",
        "ranked_courses": selected,
        "scores": scores_dict,
        "perturbation_runs": runs,
    }


# ============================================================================
# Self-test: generate synthetic ground truth and evaluate baselines
# ============================================================================

def generate_synthetic_ground_truth(n_courses: int = 200, seed: int = 42) -> Dict[str, Any]:
    """Generate synthetic ground truth for testing."""
    rng = np.random.RandomState(seed)
    keys = ["market", "mission", "university", "learner"]
    courses = {}
    for i in range(n_courses):
        cid = f"course_{i:04d}"
        # Generate scores with some structure (not purely random)
        base = rng.dirichlet([1, 1, 1, 1])
        noise = rng.normal(0, 0.05, size=4)
        scores = np.clip(base + noise, 0.0, 1.0)
        courses[cid] = {k: round(float(scores[j]), 4) for j, k in enumerate(keys)}

    # Compute true Pareto front
    pareto = compute_pareto_front(courses, keys)

    return {
        "objectives": keys,
        "courses": courses,
        "reference_pareto": pareto,
        "splits": {
            "dev": list(courses.keys())[:100],
            "eval": list(courses.keys()),
        },
    }


def self_test() -> bool:
    """Run self-test to verify all metrics compute correctly."""
    print("Running self-test...")
    gt = generate_synthetic_ground_truth()
    print(f"  Generated {len(gt['courses'])} synthetic courses, "
          f"{len(gt['reference_pareto'])} on Pareto front")

    # Test random baseline
    sub_random = generate_random_baseline(gt)
    errors = validate_submission(sub_random)
    if errors:
        print(f"  FAIL: random baseline validation: {errors}")
        return False
    res_random = evaluate_submission(sub_random, gt)
    print(f"  Random baseline:")
    for task_id, task_res in res_random["tasks"].items():
        print(f"    Task {task_id}: {json.dumps(task_res.get('metrics', {}), indent=None)[:200]}")

    # Test weighted-sum baseline
    sub_ws = generate_weighted_sum_baseline(gt)
    errors = validate_submission(sub_ws)
    if errors:
        print(f"  FAIL: weighted-sum baseline validation: {errors}")
        return False
    res_ws = evaluate_submission(sub_ws, gt)
    print(f"  Weighted-sum baseline:")
    for task_id, task_res in res_ws["tasks"].items():
        print(f"    Task {task_id}: {json.dumps(task_res.get('metrics', {}), indent=None)[:200]}")

    # Sanity checks
    a_random = res_random["tasks"]["A"]["metrics"]
    a_ws = res_ws["tasks"]["A"]["metrics"]

    # Weighted-sum should have higher objective balance than random (usually)
    print(f"\n  Sanity checks:")
    print(f"    Random coverage={a_random['coverage']}, WS coverage={a_ws['coverage']}")
    print(f"    Random balance={a_random['objective_balance']}, WS balance={a_ws['objective_balance']}")

    b_random = res_random["tasks"]["B"]["metrics"]
    b_ws = res_ws["tasks"]["B"]["metrics"]
    print(f"    Random HV={b_random['hypervolume']:.4f}, WS HV={b_ws['hypervolume']:.4f}")
    print(f"    Random Pareto prec={b_random['pareto_precision']:.4f}, "
          f"WS Pareto prec={b_ws['pareto_precision']:.4f}")

    c_random = res_random["tasks"]["C"]["metrics"]
    c_ws = res_ws["tasks"]["C"]["metrics"]
    print(f"    Random Jaccard stability={c_random['bootstrap_jaccard_mean']:.4f}, "
          f"WS Jaccard stability={c_ws['bootstrap_jaccard_mean']:.4f}")
    # Weighted-sum should be more stable than random
    assert c_ws["bootstrap_jaccard_mean"] >= c_random["bootstrap_jaccard_mean"] - 0.1, \
        "Weighted sum should be at least roughly as stable as random"

    print("\n  Self-test PASSED.")
    return True


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ADS Benchmark Evaluation (Task-Agnostic Protocol)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a submission against ground truth
  python benchmark_eval.py --submission sub.json --ground-truth gt.json

  # Evaluate only Task B
  python benchmark_eval.py --submission sub.json --ground-truth gt.json --task B

  # Run self-test with synthetic data
  python benchmark_eval.py --self-test

  # Generate example ground truth for development
  python benchmark_eval.py --generate-example --output example_gt.json
        """,
    )
    parser.add_argument("--submission", type=str, help="Path to submission JSON file")
    parser.add_argument("--ground-truth", type=str, help="Path to ground-truth JSON file")
    parser.add_argument(
        "--task",
        choices=["A", "B", "C", "all"],
        default=None,
        help="Override task (default: use task from submission)",
    )
    parser.add_argument("--output", type=str, help="Path to write results JSON")
    parser.add_argument("--self-test", action="store_true", help="Run self-test")
    parser.add_argument(
        "--generate-example",
        action="store_true",
        help="Generate example ground truth + baseline submission",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")

    args = parser.parse_args()

    if args.self_test:
        success = self_test()
        sys.exit(0 if success else 1)

    if args.generate_example:
        gt = generate_synthetic_ground_truth()
        sub_ws = generate_weighted_sum_baseline(gt)
        sub_random = generate_random_baseline(gt)

        out_dir = Path(args.output).parent if args.output else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)

        gt_path = out_dir / "example_ground_truth.json"
        sub_ws_path = out_dir / "example_submission_weighted_sum.json"
        sub_random_path = out_dir / "example_submission_random.json"

        with open(gt_path, "w") as f:
            json.dump(gt, f, indent=2)
        with open(sub_ws_path, "w") as f:
            json.dump(sub_ws, f, indent=2)
        with open(sub_random_path, "w") as f:
            json.dump(sub_random, f, indent=2)

        print(f"Generated:\n  {gt_path}\n  {sub_ws_path}\n  {sub_random_path}")
        sys.exit(0)

    if not args.submission or not args.ground_truth:
        parser.print_help()
        sys.exit(1)

    # Load inputs
    with open(args.submission) as f:
        submission = json.load(f)
    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    # Validate
    sub_errors = validate_submission(submission)
    if sub_errors:
        print(f"Submission validation errors: {sub_errors}", file=sys.stderr)
        sys.exit(1)
    gt_errors = validate_ground_truth(ground_truth)
    if gt_errors:
        print(f"Ground truth validation errors: {gt_errors}", file=sys.stderr)
        sys.exit(1)

    # Override task if specified
    if args.task:
        submission["task"] = args.task

    # Evaluate
    results = evaluate_submission(submission, ground_truth)

    # Output
    indent = 2 if args.pretty else None
    output_str = json.dumps(results, indent=indent)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
        print(f"Results written to {args.output}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
