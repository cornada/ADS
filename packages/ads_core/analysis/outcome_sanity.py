"""Outcome sanity checking: comparing predicted market-fit to observed outcomes.

This module validates ADS predictions by comparing:
1. Predicted market-fit distribution (from course/program embeddings)
2. Observed career outcomes (from FDS/outcomes surveys)

Key Metrics:
- Spearman correlation: Does the predicted ranking match observed ranking?
- MAE: How close are the predicted distributions to observed?
- Coverage: What fraction of units have usable outcomes data?

Baselines:
- B0: Keyword matching (major name only)
- B1: Content centroid (course text embeddings)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import numpy as np
from scipy import stats

from ads_core.outcomes.normalize import CanonicalOutcome, load_normalized_outcomes
from ads_core.market.targets import (
    MARKET_CATEGORIES,
    create_market_target_vectors,
    predict_market_distribution,
    keyword_baseline_prediction,
)


@dataclass
class UnitSanityResult:
    """Sanity check result for a single unit (major/program)."""

    unit: str
    unit_name: str
    institution: str
    cohort_year: str

    # Observed outcomes
    observed_employment: Optional[float] = None
    observed_grad_school: Optional[float] = None
    observed_career_outcomes: Optional[float] = None
    observed_salary: Optional[float] = None

    # Predicted market distribution (top category)
    predicted_top_category: str = ""
    predicted_top_score: float = 0.0

    # Full predicted distribution
    predicted_distribution: Dict[str, float] = field(default_factory=dict)

    # Comparison metrics
    correlation_with_salary: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "unit": self.unit,
            "unit_name": self.unit_name,
            "institution": self.institution,
            "cohort_year": self.cohort_year,
            "observed_employment": self.observed_employment,
            "observed_grad_school": self.observed_grad_school,
            "observed_career_outcomes": self.observed_career_outcomes,
            "observed_salary": self.observed_salary,
            "predicted_top_category": self.predicted_top_category,
            "predicted_top_score": self.predicted_top_score,
        }


@dataclass
class OutcomeSanityResult:
    """Aggregated sanity check results for a dataset."""

    dataset_id: str
    method: str  # "embedding", "keyword", "centroid"
    n_units: int
    n_with_outcomes: int
    coverage: float  # n_with_outcomes / n_units

    # Correlation metrics
    spearman_employment: Optional[float] = None
    spearman_salary: Optional[float] = None
    spearman_p_value: Optional[float] = None

    # Distribution metrics
    mae_employment: Optional[float] = None
    mae_salary_rank: Optional[float] = None

    # Per-unit results
    unit_results: List[UnitSanityResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "method": self.method,
            "n_units": self.n_units,
            "n_with_outcomes": self.n_with_outcomes,
            "coverage": round(self.coverage, 4),
            "spearman_employment": self.spearman_employment,
            "spearman_salary": self.spearman_salary,
            "spearman_p_value": self.spearman_p_value,
            "mae_employment": self.mae_employment,
            "mae_salary_rank": self.mae_salary_rank,
        }


def _build_unit_representations(
    outcomes: List[CanonicalOutcome],
    artifacts: List[Any],
    encoder_fn: Callable[[List[str]], np.ndarray],
    method: str = "centroid",
) -> Dict[str, np.ndarray]:
    """Build embedding representations for each unit.

    Args:
        outcomes: List of canonical outcomes
        artifacts: List of artifacts from the dataset
        encoder_fn: Text encoder function
        method: "centroid" (course text) or "name" (unit name only)

    Returns:
        Dict mapping unit to embedding vector
    """
    unit_vectors = {}

    # Group artifacts by institution-relevant categorization
    # For UCB: map majors to courses via name matching
    # For ASU: map colleges to courses via college prefix

    if method == "name":
        # Simple: embed the unit name directly
        for outcome in outcomes:
            texts = [outcome.unit_name]
            vec = encoder_fn(texts)[0]
            unit_vectors[outcome.unit] = vec

    elif method == "centroid":
        # Build centroids from course descriptions
        # This requires matching outcomes to courses

        # Extract unique units
        units = {o.unit: o for o in outcomes}

        for unit, outcome in units.items():
            # Find relevant courses based on unit name
            # This is a heuristic - match based on keywords
            unit_lower = outcome.unit_name.lower()
            relevant_texts = []

            for art in artifacts:
                # Check if artifact text is relevant to this unit
                art_text = art.text.lower() if hasattr(art, "text") else ""

                # Simple keyword matching for relevance
                # Extract key terms from unit name
                unit_terms = unit_lower.split()
                if any(term in art_text for term in unit_terms if len(term) > 3):
                    relevant_texts.append(art.text)

            if relevant_texts:
                # Compute centroid of relevant courses
                vecs = encoder_fn(relevant_texts)
                centroid = vecs.mean(axis=0)
                # Normalize
                norm = np.linalg.norm(centroid)
                if norm > 1e-8:
                    centroid = centroid / norm
                unit_vectors[unit] = centroid
            else:
                # Fall back to unit name if no courses found
                vec = encoder_fn([outcome.unit_name])[0]
                unit_vectors[unit] = vec

    return unit_vectors


def compute_outcome_sanity(
    dataset_id: str,
    outcomes: List[CanonicalOutcome],
    unit_vectors: Dict[str, np.ndarray],
    market_vectors: Dict[str, np.ndarray],
    method: str = "embedding",
) -> OutcomeSanityResult:
    """Compute outcome sanity metrics for a dataset.

    Args:
        dataset_id: Dataset identifier
        outcomes: List of canonical outcomes
        unit_vectors: Dict mapping unit to embedding vector
        market_vectors: Dict mapping category to embedding vector
        method: Method name for reporting

    Returns:
        OutcomeSanityResult with all metrics
    """
    unit_results = []
    predicted_scores = []
    observed_employment = []
    observed_salaries = []

    for outcome in outcomes:
        unit = outcome.unit

        # Skip if we don't have a vector for this unit
        if unit not in unit_vectors:
            continue

        unit_vec = unit_vectors[unit]

        # Predict market distribution
        pred_dist = predict_market_distribution(unit_vec, market_vectors)

        # Get top predicted category
        top_cat = max(pred_dist, key=pred_dist.get)
        top_score = pred_dist[top_cat]

        result = UnitSanityResult(
            unit=unit,
            unit_name=outcome.unit_name,
            institution=outcome.institution,
            cohort_year=outcome.cohort_year,
            observed_employment=outcome.employment_rate,
            observed_grad_school=outcome.grad_school_rate,
            observed_career_outcomes=outcome.career_outcomes_rate,
            observed_salary=outcome.median_salary,
            predicted_top_category=top_cat,
            predicted_top_score=top_score,
            predicted_distribution=pred_dist,
        )
        unit_results.append(result)

        # Collect for correlation
        # Use top score as predicted "market fit"
        predicted_scores.append(top_score)

        if outcome.employment_rate is not None:
            observed_employment.append(outcome.employment_rate)
        else:
            observed_employment.append(np.nan)

        if outcome.median_salary is not None:
            observed_salaries.append(outcome.median_salary)
        else:
            observed_salaries.append(np.nan)

    # Compute Spearman correlations
    pred_arr = np.array(predicted_scores)
    emp_arr = np.array(observed_employment)
    sal_arr = np.array(observed_salaries)

    # Filter out NaN values
    valid_emp = ~np.isnan(emp_arr)
    valid_sal = ~np.isnan(sal_arr)

    spearman_emp = None
    spearman_sal = None
    spearman_p = None

    if valid_emp.sum() >= 3:
        try:
            rho, p = stats.spearmanr(pred_arr[valid_emp], emp_arr[valid_emp])
            spearman_emp = round(float(rho), 4)
            spearman_p = round(float(p), 4)
        except Exception:
            pass

    if valid_sal.sum() >= 3:
        try:
            rho, _ = stats.spearmanr(pred_arr[valid_sal], sal_arr[valid_sal])
            spearman_sal = round(float(rho), 4)
        except Exception:
            pass

    # Compute MAE for employment
    mae_emp = None
    if valid_emp.sum() >= 2:
        # Normalize predicted to same scale as employment (0-1)
        # Use top_score as proxy for "market fit"
        mae_emp = round(float(np.mean(np.abs(pred_arr[valid_emp] - emp_arr[valid_emp]))), 4)

    # Coverage
    n_units = len(outcomes)
    n_with_outcomes = len(unit_results)
    coverage = n_with_outcomes / max(n_units, 1)

    return OutcomeSanityResult(
        dataset_id=dataset_id,
        method=method,
        n_units=n_units,
        n_with_outcomes=n_with_outcomes,
        coverage=round(coverage, 4),
        spearman_employment=spearman_emp,
        spearman_salary=spearman_sal,
        spearman_p_value=spearman_p,
        mae_employment=mae_emp,
        unit_results=unit_results,
    )


def compute_keyword_baseline(
    dataset_id: str,
    outcomes: List[CanonicalOutcome],
    market_vectors: Dict[str, np.ndarray],
) -> OutcomeSanityResult:
    """Compute outcome sanity using keyword baseline (B0).

    Args:
        dataset_id: Dataset identifier
        outcomes: List of canonical outcomes
        market_vectors: Dict mapping category to embedding vector (unused for keywords)

    Returns:
        OutcomeSanityResult with keyword baseline metrics
    """
    unit_results = []
    predicted_scores = []
    observed_employment = []
    observed_salaries = []

    for outcome in outcomes:
        # Predict using keywords only
        pred_dist = keyword_baseline_prediction(outcome.unit_name)

        # Get top predicted category
        top_cat = max(pred_dist, key=pred_dist.get)
        top_score = pred_dist[top_cat]

        result = UnitSanityResult(
            unit=outcome.unit,
            unit_name=outcome.unit_name,
            institution=outcome.institution,
            cohort_year=outcome.cohort_year,
            observed_employment=outcome.employment_rate,
            observed_grad_school=outcome.grad_school_rate,
            observed_career_outcomes=outcome.career_outcomes_rate,
            observed_salary=outcome.median_salary,
            predicted_top_category=top_cat,
            predicted_top_score=top_score,
            predicted_distribution=pred_dist,
        )
        unit_results.append(result)

        predicted_scores.append(top_score)

        if outcome.employment_rate is not None:
            observed_employment.append(outcome.employment_rate)
        else:
            observed_employment.append(np.nan)

        if outcome.median_salary is not None:
            observed_salaries.append(outcome.median_salary)
        else:
            observed_salaries.append(np.nan)

    # Compute correlations
    pred_arr = np.array(predicted_scores)
    emp_arr = np.array(observed_employment)
    sal_arr = np.array(observed_salaries)

    valid_emp = ~np.isnan(emp_arr)
    valid_sal = ~np.isnan(sal_arr)

    spearman_emp = None
    spearman_sal = None
    spearman_p = None

    if valid_emp.sum() >= 3:
        try:
            rho, p = stats.spearmanr(pred_arr[valid_emp], emp_arr[valid_emp])
            spearman_emp = round(float(rho), 4)
            spearman_p = round(float(p), 4)
        except Exception:
            pass

    if valid_sal.sum() >= 3:
        try:
            rho, _ = stats.spearmanr(pred_arr[valid_sal], sal_arr[valid_sal])
            spearman_sal = round(float(rho), 4)
        except Exception:
            pass

    mae_emp = None
    if valid_emp.sum() >= 2:
        mae_emp = round(float(np.mean(np.abs(pred_arr[valid_emp] - emp_arr[valid_emp]))), 4)

    n_units = len(outcomes)
    n_with_outcomes = len(unit_results)
    coverage = n_with_outcomes / max(n_units, 1)

    return OutcomeSanityResult(
        dataset_id=dataset_id,
        method="keyword_baseline",
        n_units=n_units,
        n_with_outcomes=n_with_outcomes,
        coverage=round(coverage, 4),
        spearman_employment=spearman_emp,
        spearman_salary=spearman_sal,
        spearman_p_value=spearman_p,
        mae_employment=mae_emp,
        unit_results=unit_results,
    )


def aggregate_sanity_results(
    results: List[OutcomeSanityResult],
) -> Dict[str, Any]:
    """Aggregate sanity results across datasets and methods.

    Args:
        results: List of OutcomeSanityResult

    Returns:
        Aggregated summary dict
    """
    summary = {
        "n_datasets": len(set(r.dataset_id for r in results)),
        "methods": list(set(r.method for r in results)),
        "total_units": sum(r.n_units for r in results),
        "total_with_outcomes": sum(r.n_with_outcomes for r in results),
        "results": [r.to_dict() for r in results],
    }

    # Compute mean metrics by method
    method_metrics = {}
    for method in summary["methods"]:
        method_results = [r for r in results if r.method == method]
        emp_corrs = [r.spearman_employment for r in method_results if r.spearman_employment is not None]
        sal_corrs = [r.spearman_salary for r in method_results if r.spearman_salary is not None]

        method_metrics[method] = {
            "n_datasets": len(method_results),
            "mean_spearman_employment": round(np.mean(emp_corrs), 4) if emp_corrs else None,
            "mean_spearman_salary": round(np.mean(sal_corrs), 4) if sal_corrs else None,
            "mean_coverage": round(np.mean([r.coverage for r in method_results]), 4),
        }

    summary["by_method"] = method_metrics

    return summary
