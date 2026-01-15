"""Objective functions for multi-stakeholder evaluation.

This module defines how options (e.g., courses) are scored against stakeholder targets.

Objective Convention:
- All objectives return SIMILARITY scores (higher is better, range ~[0, 1])
- This is more intuitive: "90% aligned with market" vs "0.1 distance to market"
- Pareto optimization maximizes all objectives

The four standard ADS objectives:
1. market: Alignment with labor market demands (job roles)
2. mission: Alignment with institutional mission statements
3. university: Alignment with existing curriculum
4. learner: Alignment with individual learner goals

Each objective can be computed with or without a lens transformation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from ads_core.eval.distances import cosine_distance, cosine_similarity
from ads_core.lenses.base import Lens


# ============================================================================
# Core Functions (backward compatible)
# ============================================================================

def centroid(vectors: List[np.ndarray]) -> np.ndarray:
    """Compute normalized centroid of vectors.

    Args:
        vectors: List of embedding vectors

    Returns:
        Normalized mean vector

    Raises:
        ValueError: If vectors list is empty
    """
    if len(vectors) == 0:
        raise ValueError("Empty centroid input")
    m = np.stack(vectors, axis=0).mean(axis=0)
    norm = np.linalg.norm(m)
    if norm < 1e-8:
        return m
    return m / norm


def evaluate_option(
    option_v: np.ndarray,
    target_centroids: Dict[str, np.ndarray],
    lenses: Dict[str, Lens],
) -> Dict[str, float]:
    """Evaluate an option against target centroids (backward compatible).

    NOTE: This returns SIMILARITY scores (higher is better) for consistency
    with the new objective convention.

    Args:
        option_v: Option embedding vector
        target_centroids: Dict of target name -> centroid vector
        lenses: Dict of target name -> lens transformation

    Returns:
        Dict of target name -> similarity score [0, 1]
    """
    out: Dict[str, float] = {}
    for k, t in target_centroids.items():
        lens = lenses[k]
        # Transform both option and target through the lens
        option_transformed = lens.transform(option_v)
        target_transformed = lens.transform(t)
        # Return similarity (higher is better)
        out[k] = cosine_similarity(option_transformed, target_transformed)
    return out


# ============================================================================
# Enhanced Objective Evaluation
# ============================================================================

@dataclass
class ObjectiveResult:
    """Result of evaluating a single objective."""

    name: str
    similarity: float  # Higher is better, ~[0, 1]
    distance: float    # Lower is better, [0, 2]
    raw_score: float   # Unnormalized score (for debugging)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "similarity": round(self.similarity, 6),
            "distance": round(self.distance, 6),
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for an option."""

    option_id: str
    objectives: Dict[str, float]  # name -> similarity score
    distances: Dict[str, float]   # name -> distance score
    feasible: bool = True
    violations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "option_id": self.option_id,
            "objectives": {k: round(v, 6) for k, v in self.objectives.items()},
            "feasible": self.feasible,
            "violations": self.violations,
            "metadata": self.metadata,
        }

    def to_row(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Convert to flat row for CSV export."""
        if keys is None:
            keys = list(self.objectives.keys())
        row = {"option_id": self.option_id, "feasible": self.feasible}
        for k in keys:
            row[k] = self.objectives.get(k, None)
        return row


class ObjectiveEvaluator:
    """Evaluator for multi-objective option scoring.

    This class provides a clean interface for evaluating options against
    multiple stakeholder targets with optional lens transformations.
    """

    def __init__(
        self,
        targets: Dict[str, np.ndarray],
        lenses: Optional[Dict[str, Lens]] = None,
    ):
        """Initialize evaluator.

        Args:
            targets: Dict of target name -> centroid vector
            lenses: Optional dict of target name -> lens transformation
        """
        self.targets = targets
        self.lenses = lenses or {}

    def _apply_lens(self, name: str, v: np.ndarray) -> np.ndarray:
        """Apply lens transformation if available."""
        if name in self.lenses:
            return self.lenses[name].transform(v)
        return v

    def evaluate_objective(
        self,
        option_v: np.ndarray,
        target_name: str,
    ) -> ObjectiveResult:
        """Evaluate a single objective.

        Args:
            option_v: Option embedding vector
            target_name: Name of target to evaluate against

        Returns:
            ObjectiveResult with similarity and distance scores
        """
        target = self.targets.get(target_name)
        if target is None:
            raise ValueError(f"Unknown target: {target_name}")

        # Apply lens transformations
        option_transformed = self._apply_lens(target_name, option_v)
        target_transformed = self._apply_lens(target_name, target)

        sim = cosine_similarity(option_transformed, target_transformed)
        dist = cosine_distance(option_transformed, target_transformed)

        return ObjectiveResult(
            name=target_name,
            similarity=sim,
            distance=dist,
            raw_score=sim,
        )

    def evaluate(
        self,
        option_id: str,
        option_v: np.ndarray,
        objective_names: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate an option against all or specified objectives.

        Args:
            option_id: Identifier for the option
            option_v: Option embedding vector
            objective_names: List of objectives to evaluate (default: all)

        Returns:
            EvaluationResult with all scores
        """
        if objective_names is None:
            objective_names = list(self.targets.keys())

        objectives = {}
        distances = {}

        for name in objective_names:
            if name not in self.targets:
                continue
            result = self.evaluate_objective(option_v, name)
            objectives[name] = result.similarity
            distances[name] = result.distance

        return EvaluationResult(
            option_id=option_id,
            objectives=objectives,
            distances=distances,
        )

    def evaluate_batch(
        self,
        options: Dict[str, np.ndarray],
        objective_names: Optional[List[str]] = None,
    ) -> List[EvaluationResult]:
        """Evaluate multiple options.

        Args:
            options: Dict of option_id -> embedding vector
            objective_names: List of objectives to evaluate

        Returns:
            List of EvaluationResult, one per option
        """
        results = []
        for option_id, option_v in options.items():
            result = self.evaluate(option_id, option_v, objective_names)
            results.append(result)
        return results


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_objectives(
    results: List[EvaluationResult],
    method: str = "minmax",
) -> List[EvaluationResult]:
    """Normalize objective scores across a batch.

    Args:
        results: List of evaluation results
        method: Normalization method ("minmax", "zscore", or "none")

    Returns:
        New list with normalized scores
    """
    if not results or method == "none":
        return results

    # Collect all objective names
    all_names = set()
    for r in results:
        all_names.update(r.objectives.keys())

    # Compute statistics per objective
    stats: Dict[str, Dict[str, float]] = {}
    for name in all_names:
        values = [r.objectives.get(name, 0) for r in results if name in r.objectives]
        if values:
            stats[name] = {
                "min": min(values),
                "max": max(values),
                "mean": np.mean(values),
                "std": np.std(values) + 1e-8,
            }

    # Apply normalization
    normalized = []
    for r in results:
        new_objectives = {}
        for name, value in r.objectives.items():
            if name in stats:
                s = stats[name]
                if method == "minmax":
                    range_val = s["max"] - s["min"]
                    if range_val > 1e-8:
                        new_objectives[name] = (value - s["min"]) / range_val
                    else:
                        new_objectives[name] = 0.5  # All same value
                elif method == "zscore":
                    new_objectives[name] = (value - s["mean"]) / s["std"]
                else:
                    new_objectives[name] = value
            else:
                new_objectives[name] = value

        normalized.append(EvaluationResult(
            option_id=r.option_id,
            objectives=new_objectives,
            distances=r.distances,
            feasible=r.feasible,
            violations=r.violations,
            metadata={**r.metadata, "normalization": method},
        ))

    return normalized


def objectives_to_csv_rows(
    results: List[EvaluationResult],
    objective_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Convert evaluation results to CSV-friendly rows.

    Args:
        results: List of evaluation results
        objective_names: Column order for objectives

    Returns:
        List of dicts suitable for csv.DictWriter
    """
    if not results:
        return []

    if objective_names is None:
        # Collect all names and sort for consistency
        all_names = set()
        for r in results:
            all_names.update(r.objectives.keys())
        objective_names = sorted(all_names)

    return [r.to_row(objective_names) for r in results]
