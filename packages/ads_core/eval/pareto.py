"""Pareto front computation for multi-objective optimization.

Convention: HIGHER is better for all objectives (maximization).
This matches the objective convention where similarity scores are used.

Key concepts:
- Dominance: solution A dominates B if A is at least as good in all objectives
  and strictly better in at least one
- Pareto front: set of all non-dominated solutions (optimal trade-offs)
- Pareto rank: iterative fronts (rank 0 = best, rank 1 = dominated only by rank 0, etc.)

Contestability support:
- Filter solutions by feasibility before computing fronts
- Support for objective subsets (e.g., market+mission vs all four)
- Hypervolume indicator for comparing Pareto fronts
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import numpy as np


@dataclass
class ParetoResult:
    """Result of Pareto front computation with metadata."""

    indices: List[int]
    solutions: List[Dict[str, float]]
    keys: List[str]
    total_count: int
    feasible_count: int = 0
    hypervolume: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pareto_ratio(self) -> float:
        """Fraction of solutions that are Pareto-optimal."""
        return len(self.indices) / max(self.total_count, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indices": self.indices,
            "count": len(self.indices),
            "total_count": self.total_count,
            "pareto_ratio": round(self.pareto_ratio, 4),
            "feasible_count": self.feasible_count,
            "keys": self.keys,
            "hypervolume": self.hypervolume,
            "metadata": self.metadata,
        }


def dominates(a: Dict[str, float], b: Dict[str, float], keys: Sequence[str]) -> bool:
    """Check if solution `a` dominates solution `b` (maximization).

    `a` dominates `b` if:
    - a[k] >= b[k] for all objectives k (a is at least as good)
    - a[k] > b[k] for at least one objective k (a is strictly better somewhere)

    Args:
        a: First solution's objective values
        b: Second solution's objective values
        keys: Objective names to consider

    Returns:
        True if `a` dominates `b`
    """
    at_least_as_good = all(a[k] >= b[k] for k in keys)
    strictly_better = any(a[k] > b[k] for k in keys)
    return at_least_as_good and strictly_better


def is_dominated(a: Dict[str, float], b: Dict[str, float], keys: Sequence[str]) -> bool:
    """Check if solution `a` is dominated by solution `b`.

    Backward-compatible alias: returns True if `b` dominates `a`.
    """
    return dominates(b, a, keys)


def pareto_front(items: List[Dict[str, float]], keys: Sequence[str]) -> List[int]:
    """Find indices of Pareto-optimal solutions (maximization).

    A solution is Pareto-optimal (non-dominated) if no other solution
    dominates it, i.e., no other solution is at least as good in all
    objectives and strictly better in at least one.

    Args:
        items: List of solutions, each a dict of objective name -> value
        keys: Objective names to consider

    Returns:
        List of indices into `items` that are Pareto-optimal

    Example:
        >>> items = [{"x": 0.8, "y": 0.3}, {"x": 0.5, "y": 0.6}, {"x": 0.9, "y": 0.2}]
        >>> pareto_front(items, ["x", "y"])
        [0, 1, 2]  # All non-dominated (trade-off between x and y)
    """
    nondom: List[int] = []
    n = len(items)

    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j and dominates(items[j], items[i], keys):
                dominated = True
                break
        if not dominated:
            nondom.append(i)

    return nondom


def pareto_rank(items: List[Dict[str, float]], keys: Sequence[str]) -> List[int]:
    """Compute Pareto rank for each solution.

    Rank 0 = Pareto front (best), higher ranks = dominated by lower ranks.

    Args:
        items: List of solutions
        keys: Objective names

    Returns:
        List of ranks (0 = Pareto-optimal)
    """
    n = len(items)
    if n == 0:
        return []

    ranks = [-1] * n
    remaining = set(range(n))
    current_rank = 0

    while remaining:
        # Find non-dominated among remaining
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i != j and dominates(items[j], items[i], keys):
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        # Assign rank to front
        for i in front:
            ranks[i] = current_rank
            remaining.remove(i)

        current_rank += 1

    return ranks


# ============================================================================
# Enhanced Pareto Analysis
# ============================================================================

def compute_pareto_with_feasibility(
    items: List[Dict[str, Any]],
    keys: Sequence[str],
    feasibility_key: str = "feasible",
    feasible_only: bool = True,
) -> ParetoResult:
    """Compute Pareto front with feasibility filtering.

    Args:
        items: List of solution dicts with 'objectives' and 'feasible' fields
        keys: Objective names to consider
        feasibility_key: Key for feasibility flag
        feasible_only: If True, only consider feasible solutions

    Returns:
        ParetoResult with Pareto front info
    """
    # Extract objectives and filter by feasibility
    objectives = []
    original_indices = []
    feasible_count = 0

    for i, item in enumerate(items):
        is_feasible = item.get(feasibility_key, True)
        if is_feasible:
            feasible_count += 1

        if not feasible_only or is_feasible:
            obj = item.get("objectives", item)
            objectives.append(obj)
            original_indices.append(i)

    # Compute Pareto front on filtered set
    if objectives:
        pareto_indices = pareto_front(objectives, keys)
        pareto_original = [original_indices[i] for i in pareto_indices]
        pareto_solutions = [objectives[i] for i in pareto_indices]
    else:
        pareto_original = []
        pareto_solutions = []

    return ParetoResult(
        indices=pareto_original,
        solutions=pareto_solutions,
        keys=list(keys),
        total_count=len(items),
        feasible_count=feasible_count,
        metadata={"feasible_only": feasible_only},
    )


def compute_hypervolume(
    solutions: List[Dict[str, float]],
    keys: Sequence[str],
    reference_point: Optional[Dict[str, float]] = None,
) -> float:
    """Compute hypervolume indicator for a Pareto front.

    Hypervolume measures the volume of objective space dominated by the front,
    bounded by a reference point. Higher hypervolume = better front.

    Uses a simple 2D algorithm; for >2 objectives, approximates via sampling.

    Args:
        solutions: List of Pareto-optimal solutions
        keys: Objective names
        reference_point: Reference point for hypervolume (defaults to origin)

    Returns:
        Hypervolume value (higher is better)
    """
    if not solutions:
        return 0.0

    n_obj = len(keys)
    if reference_point is None:
        reference_point = {k: 0.0 for k in keys}

    # Convert to numpy array for easier computation
    points = np.array([[s[k] for k in keys] for s in solutions])
    ref = np.array([reference_point[k] for k in keys])

    # Shift points relative to reference
    shifted = points - ref

    # Filter out points dominated by reference (negative in any dimension)
    valid_mask = np.all(shifted > 0, axis=1)
    shifted = shifted[valid_mask]

    if len(shifted) == 0:
        return 0.0

    if n_obj == 2:
        # Exact 2D hypervolume
        return _hypervolume_2d(shifted)
    else:
        # Monte Carlo approximation for higher dimensions
        return _hypervolume_monte_carlo(shifted, n_samples=10000)


def _hypervolume_2d(points: np.ndarray) -> float:
    """Exact 2D hypervolume computation."""
    # Sort by first objective descending
    sorted_idx = np.argsort(-points[:, 0])
    points = points[sorted_idx]

    hv = 0.0
    prev_y = 0.0

    for x, y in points:
        if y > prev_y:
            hv += x * (y - prev_y)
            prev_y = y

    return float(hv)


def _hypervolume_monte_carlo(points: np.ndarray, n_samples: int = 10000) -> float:
    """Monte Carlo approximation of hypervolume for n>2 objectives."""
    # Compute bounding box
    max_point = points.max(axis=0)

    # Sample random points in bounding box
    rng = np.random.RandomState(42)
    samples = rng.uniform(0, max_point, size=(n_samples, points.shape[1]))

    # Count samples dominated by at least one Pareto point
    dominated_count = 0
    for sample in samples:
        for point in points:
            if np.all(sample <= point):
                dominated_count += 1
                break

    # Hypervolume = fraction dominated * bounding box volume
    box_volume = np.prod(max_point)
    return float(dominated_count / n_samples * box_volume)


def compare_pareto_fronts(
    front_a: List[Dict[str, float]],
    front_b: List[Dict[str, float]],
    keys: Sequence[str],
) -> Dict[str, Any]:
    """Compare two Pareto fronts.

    Args:
        front_a: First Pareto front
        front_b: Second Pareto front
        keys: Objective names

    Returns:
        Comparison metrics
    """
    hv_a = compute_hypervolume(front_a, keys)
    hv_b = compute_hypervolume(front_b, keys)

    # Count how many solutions from each front dominate solutions in the other
    a_dominates_b = 0
    b_dominates_a = 0

    for sol_a in front_a:
        for sol_b in front_b:
            if dominates(sol_a, sol_b, keys):
                a_dominates_b += 1
            elif dominates(sol_b, sol_a, keys):
                b_dominates_a += 1

    return {
        "hypervolume_a": hv_a,
        "hypervolume_b": hv_b,
        "hypervolume_ratio": hv_a / max(hv_b, 1e-8),
        "a_dominates_b_count": a_dominates_b,
        "b_dominates_a_count": b_dominates_a,
        "size_a": len(front_a),
        "size_b": len(front_b),
    }
