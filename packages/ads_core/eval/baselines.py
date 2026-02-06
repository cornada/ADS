"""Scalarization baselines for multi-objective comparison.

Baselines convert multi-objective optimization into single-objective
problems through scalarization. These serve as comparisons for
Pareto-based methods.

Supported baselines:
1. Weighted Sum: Linear combination with fixed or grid weights
2. Random Scalarization: Sample weight vectors from simplex
3. Lexicographic: Prioritize objectives in order

Convention: All baselines return solutions ordered by their scalar score
(higher is better), matching the MAXIMIZATION convention in pareto.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class ScalarizationResult:
    """Result of a scalarization baseline."""

    method: str
    weights: Dict[str, float]  # Objective weights used
    ranked_indices: List[int]  # Indices sorted by scalar score (best first)
    scores: List[float]  # Scalar scores for each item
    top_k: List[int]  # Indices of top-k solutions
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "top_k": self.top_k,
            "top_k_count": len(self.top_k),
            "metadata": self.metadata,
        }


@dataclass
class BaselineComparisonResult:
    """Result of comparing baselines to Pareto front."""

    baseline_method: str
    baseline_top_k: List[int]
    pareto_indices: List[int]
    overlap_count: int  # |baseline_top_k ∩ pareto_indices|
    overlap_ratio: float  # overlap_count / |pareto_indices|
    coverage_ratio: float  # overlap_count / |baseline_top_k|
    unique_to_baseline: List[int]
    unique_to_pareto: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.baseline_method,
            "overlap_count": self.overlap_count,
            "overlap_ratio": round(self.overlap_ratio, 4),
            "coverage_ratio": round(self.coverage_ratio, 4),
            "unique_to_baseline": len(self.unique_to_baseline),
            "unique_to_pareto": len(self.unique_to_pareto),
        }


# ============================================================================
# Weighted Sum Scalarization
# ============================================================================

def weighted_sum(
    items: List[Dict[str, float]],
    keys: Sequence[str],
    weights: Dict[str, float],
    top_k: int = 10,
) -> ScalarizationResult:
    """Scalarize objectives using weighted sum.

    Score(x) = Σ w_i * x_i

    Args:
        items: List of solutions with objective values
        keys: Objective names to consider
        weights: Dict of objective name -> weight
        top_k: Number of top solutions to return

    Returns:
        ScalarizationResult with ranked solutions
    """
    # Normalize weights to sum to 1
    keys = list(keys)
    if len(keys) == 0:
        return ScalarizationResult(
            method="weighted_sum",
            weights={},
            ranked_indices=list(range(len(items))),
            scores=[0.0] * len(items),
            top_k=list(range(min(top_k, len(items)))),
            metadata={"top_k_requested": top_k, "warning": "empty keys"},
        )

    total = sum(weights.get(k, 0) for k in keys)
    if total < 1e-8:
        norm_weights = {k: 1.0 / len(keys) for k in keys}
    else:
        norm_weights = {k: weights.get(k, 0) / total for k in keys}

    # Compute scalar scores
    scores = []
    for item in items:
        score = sum(norm_weights[k] * item.get(k, 0) for k in keys)
        scores.append(score)

    # Rank by score (descending)
    ranked = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
    top = ranked[:top_k]

    return ScalarizationResult(
        method="weighted_sum",
        weights=norm_weights,
        ranked_indices=ranked,
        scores=scores,
        top_k=top,
        metadata={"top_k_requested": top_k},
    )


def weighted_sum_grid(
    items: List[Dict[str, float]],
    keys: Sequence[str],
    grid_resolution: int = 5,
    top_k: int = 10,
) -> List[ScalarizationResult]:
    """Run weighted sum with grid of weight combinations.

    For 2 objectives: weights from [0, 0.25, 0.5, 0.75, 1.0]
    For N objectives: enumerate combinations summing to 1

    Args:
        items: List of solutions
        keys: Objective names
        grid_resolution: Number of points per dimension (default 5)
        top_k: Top solutions per weight combination

    Returns:
        List of ScalarizationResult, one per weight combination
    """
    keys = list(keys)
    n_obj = len(keys)

    # Generate weight grid
    weight_values = [i / (grid_resolution - 1) for i in range(grid_resolution)]

    results = []
    if n_obj == 2:
        # Simple 1D grid for 2 objectives
        for w in weight_values:
            weights = {keys[0]: w, keys[1]: 1.0 - w}
            result = weighted_sum(items, keys, weights, top_k)
            results.append(result)
    else:
        # Generate all combinations summing to 1 for N objectives
        weight_combos = _generate_weight_simplex(n_obj, grid_resolution)
        for combo in weight_combos:
            weights = {keys[i]: combo[i] for i in range(n_obj)}
            result = weighted_sum(items, keys, weights, top_k)
            results.append(result)

    return results


def _generate_weight_simplex(n_dims: int, resolution: int) -> List[Tuple[float, ...]]:
    """Generate weight vectors on simplex (sum to 1).

    Uses recursive approach to enumerate all combinations.
    """
    if n_dims == 1:
        return [(1.0,)]

    combos = []
    for i in range(resolution):
        w_first = i / (resolution - 1)
        remaining = 1.0 - w_first

        if n_dims == 2:
            combos.append((w_first, remaining))
        else:
            # Recursively generate for remaining dimensions
            sub_combos = _generate_weight_simplex(n_dims - 1, resolution)
            for sub in sub_combos:
                scaled = tuple(w * remaining for w in sub)
                combos.append((w_first,) + scaled)

    return combos


# ============================================================================
# Random Scalarization
# ============================================================================

def random_scalarization(
    items: List[Dict[str, float]],
    keys: Sequence[str],
    n_samples: int = 100,
    top_k: int = 10,
    seed: int = 42,
) -> List[ScalarizationResult]:
    """Scalarize with randomly sampled weight vectors.

    Samples weights uniformly from the simplex (Dirichlet with alpha=1).

    Args:
        items: List of solutions
        keys: Objective names
        n_samples: Number of random weight vectors
        top_k: Top solutions per sample
        seed: Random seed

    Returns:
        List of ScalarizationResult, one per weight sample
    """
    keys = list(keys)
    n_obj = len(keys)
    rng = np.random.RandomState(seed)

    results = []
    for _ in range(n_samples):
        # Sample from Dirichlet (uniform on simplex)
        raw = rng.exponential(1.0, size=n_obj)
        weights_arr = raw / raw.sum()
        weights = {keys[i]: float(weights_arr[i]) for i in range(n_obj)}

        result = weighted_sum(items, keys, weights, top_k)
        result.method = "random_scalarization"
        results.append(result)

    return results


def aggregate_random_scalarization(
    results: List[ScalarizationResult],
    n_items: int,
    top_k: int = 10,
) -> ScalarizationResult:
    """Aggregate random scalarization by selection frequency.

    Counts how often each item appears in top-k across samples.

    Args:
        results: List of ScalarizationResult from random samples
        n_items: Total number of items
        top_k: Number of top solutions to return

    Returns:
        ScalarizationResult with frequency-based ranking
    """
    # Count selection frequency
    freq = np.zeros(n_items)
    for r in results:
        for idx in r.top_k:
            freq[idx] += 1

    # Normalize by number of samples
    freq = freq / len(results)

    # Rank by frequency
    ranked = sorted(range(n_items), key=lambda i: freq[i], reverse=True)
    top = ranked[:top_k]

    return ScalarizationResult(
        method="random_scalarization_aggregate",
        weights={},  # Aggregated, no single weight vector
        ranked_indices=ranked,
        scores=freq.tolist(),
        top_k=top,
        metadata={
            "n_samples": len(results),
            "top_k_requested": top_k,
        },
    )


# ============================================================================
# Lexicographic Ordering
# ============================================================================

def lexicographic(
    items: List[Dict[str, float]],
    keys: Sequence[str],
    priority_order: Optional[List[str]] = None,
    epsilon: float = 0.01,
    top_k: int = 10,
) -> ScalarizationResult:
    """Lexicographic ordering of objectives.

    Prioritizes objectives in order. For each objective, only consider
    items within epsilon of the best value, then proceed to next objective.

    Args:
        items: List of solutions
        keys: Objective names (used as priority if priority_order not given)
        priority_order: Objectives in priority order (first = highest)
        epsilon: Tolerance for considering items "tied" on an objective
        top_k: Number of top solutions to return

    Returns:
        ScalarizationResult with lexicographic ranking
    """
    if priority_order is None:
        priority_order = list(keys)

    # Build sort key: tuple of objectives in priority order
    # Use (1 - value) so that sorting ascending gives us highest values first
    def sort_key(idx: int) -> Tuple[float, ...]:
        item = items[idx]
        return tuple(-item.get(k, 0) for k in priority_order)

    # Rank items
    ranked = sorted(range(len(items)), key=sort_key)

    # Compute pseudo-scores (sum weighted by position in priority)
    scores = []
    n_obj = len(priority_order)
    for item in items:
        score = sum(
            (n_obj - i) * item.get(k, 0)
            for i, k in enumerate(priority_order)
        )
        scores.append(score)

    top = ranked[:top_k]

    # Weights represent priority (higher priority = higher weight)
    weights = {k: float(n_obj - i) for i, k in enumerate(priority_order)}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return ScalarizationResult(
        method="lexicographic",
        weights=weights,
        ranked_indices=ranked,
        scores=scores,
        top_k=top,
        metadata={
            "priority_order": priority_order,
            "epsilon": epsilon,
            "top_k_requested": top_k,
        },
    )


# ============================================================================
# Epsilon-Constraint Method
# ============================================================================

def epsilon_constraint(
    items: List[Dict[str, float]],
    keys: Sequence[str],
    primary: str,
    constraints: Dict[str, float],
    top_k: int = 10,
) -> ScalarizationResult:
    """Epsilon-constraint scalarization.

    Optimize primary objective subject to minimum thresholds on others.

    Args:
        items: List of solutions
        keys: Objective names
        primary: Primary objective to maximize
        constraints: Dict of objective -> minimum threshold
        top_k: Top solutions to return

    Returns:
        ScalarizationResult with feasible solutions ranked by primary
    """
    # Filter by constraints
    feasible_indices = []
    for i, item in enumerate(items):
        satisfies = all(
            item.get(obj, 0) >= threshold
            for obj, threshold in constraints.items()
        )
        if satisfies:
            feasible_indices.append(i)

    # Rank feasible items by primary objective
    if feasible_indices:
        feasible_indices.sort(key=lambda i: items[i].get(primary, 0), reverse=True)

    # Compute scores (primary objective value, 0 for infeasible)
    scores = []
    for i, item in enumerate(items):
        if i in feasible_indices:
            scores.append(item.get(primary, 0))
        else:
            scores.append(0.0)

    # Full ranking: feasible first (by primary), then infeasible
    infeasible = [i for i in range(len(items)) if i not in feasible_indices]
    ranked = feasible_indices + infeasible

    top = feasible_indices[:top_k]

    # Weights: 1 for primary, 0 for others (constraints not weighted)
    weights = {k: 1.0 if k == primary else 0.0 for k in keys}

    return ScalarizationResult(
        method="epsilon_constraint",
        weights=weights,
        ranked_indices=ranked,
        scores=scores,
        top_k=top,
        metadata={
            "primary": primary,
            "constraints": constraints,
            "feasible_count": len(feasible_indices),
            "top_k_requested": top_k,
        },
    )


# ============================================================================
# Baseline Comparison
# ============================================================================

def compare_to_pareto(
    baseline_result: ScalarizationResult,
    pareto_indices: List[int],
) -> BaselineComparisonResult:
    """Compare baseline selection to Pareto front.

    Args:
        baseline_result: Result from a baseline method
        pareto_indices: Indices of Pareto-optimal solutions

    Returns:
        BaselineComparisonResult with overlap metrics
    """
    baseline_set = set(baseline_result.top_k)
    pareto_set = set(pareto_indices)

    overlap = baseline_set & pareto_set
    unique_baseline = baseline_set - pareto_set
    unique_pareto = pareto_set - baseline_set

    return BaselineComparisonResult(
        baseline_method=baseline_result.method,
        baseline_top_k=baseline_result.top_k,
        pareto_indices=pareto_indices,
        overlap_count=len(overlap),
        overlap_ratio=len(overlap) / max(len(pareto_set), 1),
        coverage_ratio=len(overlap) / max(len(baseline_set), 1),
        unique_to_baseline=list(unique_baseline),
        unique_to_pareto=list(unique_pareto),
        metadata={"baseline_weights": baseline_result.weights},
    )


def run_all_baselines(
    items: List[Dict[str, float]],
    keys: Sequence[str],
    top_k: int = 10,
    grid_resolution: int = 5,
    n_random_samples: int = 50,
    seed: int = 42,
) -> Dict[str, ScalarizationResult]:
    """Run all baseline methods.

    Args:
        items: List of solutions
        keys: Objective names
        top_k: Top solutions per method
        grid_resolution: Grid resolution for weighted sum
        n_random_samples: Number of random weight samples
        seed: Random seed

    Returns:
        Dict of method name -> ScalarizationResult
    """
    keys = list(keys)
    results = {}

    # 1. Equal weights
    equal_weights = {k: 1.0 / len(keys) for k in keys}
    results["weighted_sum_equal"] = weighted_sum(items, keys, equal_weights, top_k)

    # 2. Weighted sum grid (aggregate best performers)
    grid_results = weighted_sum_grid(items, keys, grid_resolution, top_k)
    # Use the result that captures most unique solutions
    unique_union = set()
    for r in grid_results:
        unique_union.update(r.top_k)
    results["weighted_sum_grid"] = ScalarizationResult(
        method="weighted_sum_grid",
        weights={},
        ranked_indices=[],  # Multiple weight vectors, no single ranking
        scores=[],
        top_k=list(unique_union)[:top_k],
        metadata={"n_weight_combos": len(grid_results)},
    )

    # 3. Random scalarization (aggregate)
    random_results = random_scalarization(items, keys, n_random_samples, top_k, seed)
    results["random_scalarization"] = aggregate_random_scalarization(
        random_results, len(items), top_k
    )

    # 4. Lexicographic orderings (one per objective as primary)
    for primary in keys:
        priority = [primary] + [k for k in keys if k != primary]
        lex_result = lexicographic(items, keys, priority, top_k=top_k)
        results[f"lexicographic_{primary}"] = lex_result

    return results
