"""Constraint functions for contestable multi-objective optimization.

Constraints enable stakeholders to express preferences beyond objectives:
- Autonomy drift: ensure learner agency isn't overridden by market pressure
- Minimum thresholds: require minimum alignment with certain objectives
- Diversity: require coverage across different objective directions

Contestability:
- tau parameter allows adjusting constraint strictness
- Constraints can be enabled/disabled via config
- Different stakeholders can have different constraint profiles

Autonomy Drift Formalization:
The autonomy drift metric D(r) measures how much a recommendation r diverges
from the learner's stated preferences toward external stakeholder objectives.

D(r) = w_m * max(0, S_m(r) - S_l(r)) + w_u * max(0, S_u(r) - S_l(r))

Where:
- S_l(r) = learner similarity score
- S_m(r) = market similarity score
- S_u(r) = university similarity score
- w_m, w_u = weights for each drift component (default: 0.5, 0.5)

Interpretation:
- D(r) = 0: recommendation fully aligned with learner preferences
- D(r) > 0: recommendation prioritizes external objectives over learner
- D(r) > tau: constraint violation (recommendation rejected or flagged)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np


@dataclass
class ConstraintResult:
    """Result of constraint evaluation."""

    feasible: bool
    violations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible": self.feasible,
            "violations": self.violations,
            "metadata": self.metadata,
        }


@dataclass
class ConstraintConfig:
    """Configuration for constraint evaluation.

    Attributes:
        enable_autonomy: Enable autonomy drift constraint
        autonomy_tau: Threshold for autonomy drift
        min_thresholds: Minimum required values for objectives
        max_thresholds: Maximum allowed values for objectives
    """

    enable_autonomy: bool = True
    autonomy_tau: float = 0.25
    min_thresholds: Dict[str, float] = field(default_factory=dict)
    max_thresholds: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConstraintConfig":
        return cls(
            enable_autonomy=d.get("enable_ethics", d.get("enable_autonomy", True)),
            autonomy_tau=d.get("autonomy_drift_tau", 0.25),
            min_thresholds=d.get("min_thresholds", {}),
            max_thresholds=d.get("max_thresholds", {}),
        )


def autonomy_drift(
    objectives: Dict[str, float],
    weight_market: float = 0.5,
    weight_university: float = 0.5,
) -> float:
    """Compute autonomy drift metric D(r).

    Formal definition:
    D(r) = w_m * max(0, S_m(r) - S_l(r)) + w_u * max(0, S_u(r) - S_l(r))

    This measures how much a recommendation prioritizes external stakeholder
    objectives (market, university) over the learner's stated preferences.

    Args:
        objectives: Dict of objective name -> similarity value
        weight_market: Weight for market drift component (default 0.5)
        weight_university: Weight for university drift component (default 0.5)

    Returns:
        Drift value D(r) in [0, 1]:
        - 0 = perfectly aligned with learner
        - > 0 = prioritizes external objectives
        - 1 = maximum drift (external fully dominates learner)
    """
    learner = objectives.get("learner", 0.0)
    market = objectives.get("market", 0.0)
    university = objectives.get("university", 0.0)

    # Compute drift components
    market_drift = max(0.0, market - learner)
    university_drift = max(0.0, university - learner)

    # Weighted sum
    drift = weight_market * market_drift + weight_university * university_drift

    return drift


def autonomy_drift_detailed(objectives: Dict[str, float]) -> Dict[str, float]:
    """Compute detailed autonomy drift breakdown.

    Returns individual drift components for interpretability.

    Args:
        objectives: Dict of objective values

    Returns:
        Dict with total drift and per-stakeholder breakdown
    """
    learner = objectives.get("learner", 0.0)
    market = objectives.get("market", 0.0)
    university = objectives.get("university", 0.0)
    mission = objectives.get("mission", 0.0)

    return {
        "total": autonomy_drift(objectives),
        "market_drift": max(0.0, market - learner),
        "university_drift": max(0.0, university - learner),
        "mission_drift": max(0.0, mission - learner),
        "learner_score": learner,
        "market_score": market,
        "university_score": university,
    }


def apply_constraints(objectives: Dict[str, float], tau: float) -> Dict[str, Any]:
    """Apply autonomy drift constraint (backward compatible).

    Args:
        objectives: Dict of objective values
        tau: Autonomy drift threshold

    Returns:
        Dict with feasible flag and violations
    """
    drift = autonomy_drift(objectives)
    feasible = drift <= tau
    return {
        "feasible": feasible,
        "violations": {} if feasible else {"autonomy_drift": drift, "tau": tau},
    }


def evaluate_constraints(
    objectives: Dict[str, float],
    config: ConstraintConfig,
) -> ConstraintResult:
    """Evaluate all constraints with full configuration.

    Args:
        objectives: Dict of objective values
        config: Constraint configuration

    Returns:
        ConstraintResult with all violations
    """
    violations: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {"config": {"tau": config.autonomy_tau}}

    # Autonomy drift constraint
    if config.enable_autonomy:
        drift = autonomy_drift(objectives)
        if drift > config.autonomy_tau:
            violations["autonomy_drift"] = {
                "value": drift,
                "threshold": config.autonomy_tau,
                "exceeded_by": drift - config.autonomy_tau,
            }

    # Minimum threshold constraints
    for obj_name, min_val in config.min_thresholds.items():
        actual = objectives.get(obj_name, 0.0)
        if actual < min_val:
            violations[f"min_{obj_name}"] = {
                "value": actual,
                "threshold": min_val,
                "deficit": min_val - actual,
            }

    # Maximum threshold constraints
    for obj_name, max_val in config.max_thresholds.items():
        actual = objectives.get(obj_name, 0.0)
        if actual > max_val:
            violations[f"max_{obj_name}"] = {
                "value": actual,
                "threshold": max_val,
                "exceeded_by": actual - max_val,
            }

    feasible = len(violations) == 0
    return ConstraintResult(feasible=feasible, violations=violations, metadata=metadata)


def create_constraint_fn(config: ConstraintConfig) -> Callable[[Dict[str, float]], Dict[str, Any]]:
    """Create a constraint function from config.

    Args:
        config: Constraint configuration

    Returns:
        Function that takes objectives and returns constraint result dict
    """
    def constraint_fn(objectives: Dict[str, float]) -> Dict[str, Any]:
        result = evaluate_constraints(objectives, config)
        return result.to_dict()

    return constraint_fn


# ============================================================================
# Preset Constraint Profiles
# ============================================================================

def strict_autonomy_config(tau: float = 0.1) -> ConstraintConfig:
    """Strict autonomy constraints - prioritize learner agency."""
    return ConstraintConfig(enable_autonomy=True, autonomy_tau=tau)


def relaxed_autonomy_config(tau: float = 0.5) -> ConstraintConfig:
    """Relaxed autonomy constraints - allow more market influence."""
    return ConstraintConfig(enable_autonomy=True, autonomy_tau=tau)


def no_constraints_config() -> ConstraintConfig:
    """No constraints - pure Pareto optimization."""
    return ConstraintConfig(enable_autonomy=False)
