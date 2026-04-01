"""Sensitivity analysis for observational curriculum studies.

Since we cannot run randomized experiments on curricula, we use
sensitivity analysis to answer: "How strong would an unmeasured
confounder need to be to explain away our findings?"

Methods:
1. E-value (VanderWeele & Ding, 2017): minimum strength of confounding
   that could reduce the observed association to null.
2. Rosenbaum bounds: how sensitive is a matched comparison to hidden bias.
3. Manski bounds: partial identification under minimal assumptions.

Educational interpretation:
- E-value for curriculum → market_fit association tells us: "if there's
  an unmeasured factor (e.g., student self-selection) that makes both
  curriculum choice and market success more likely, it would need to
  multiply odds by E to explain away our observed effect."
- If E-value is large (>3), our finding is robust to moderate confounding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class EValueResult:
    """E-value sensitivity analysis result."""

    point_estimate: float   # observed effect size (risk ratio or similar)
    e_value: float          # E-value for the point estimate
    e_value_ci: float       # E-value for the confidence interval bound
    interpretation: str     # human-readable interpretation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RosenbaumResult:
    """Rosenbaum sensitivity analysis for matched pairs."""

    gamma_values: List[float]   # bias magnitudes tested
    p_values: List[float]       # p-values at each gamma
    critical_gamma: float       # smallest gamma where p > alpha
    interpretation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ManskiBoundsResult:
    """Manski partial identification bounds."""

    lower_bound: float   # worst-case lower bound on treatment effect
    upper_bound: float   # worst-case upper bound
    point_estimate: float
    selection_fraction: float  # fraction of data affected by selection
    interpretation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# E-value
# ============================================================================

def compute_e_value(
    risk_ratio: float,
    ci_lower: Optional[float] = None,
) -> EValueResult:
    """Compute E-value for a risk ratio.

    The E-value is the minimum strength of association that an
    unmeasured confounder would need to have with both treatment and
    outcome to fully explain away the observed association.

    Args:
        risk_ratio: Observed risk ratio (must be >= 1; if < 1, invert)
        ci_lower: Lower bound of confidence interval for the risk ratio

    Returns:
        EValueResult with E-value and interpretation
    """
    if risk_ratio < 1:
        risk_ratio = 1.0 / risk_ratio

    # E-value formula (VanderWeele & Ding, 2017)
    e_value = risk_ratio + np.sqrt(risk_ratio * (risk_ratio - 1))

    # E-value for CI bound
    e_value_ci = 0.0
    if ci_lower is not None:
        if ci_lower < 1:
            ci_lower = 1.0 / ci_lower
        if ci_lower > 1:
            e_value_ci = ci_lower + np.sqrt(ci_lower * (ci_lower - 1))
        else:
            e_value_ci = 1.0

    # Interpretation
    if e_value > 4:
        strength = "very robust"
    elif e_value > 2.5:
        strength = "moderately robust"
    elif e_value > 1.5:
        strength = "somewhat fragile"
    else:
        strength = "fragile"

    interpretation = (
        f"E-value = {e_value:.2f}: an unmeasured confounder would need to "
        f"multiply both treatment-confounder and confounder-outcome associations "
        f"by at least {e_value:.2f} to explain away the observed RR={risk_ratio:.2f}. "
        f"Finding is {strength} to unmeasured confounding."
    )

    return EValueResult(
        point_estimate=risk_ratio,
        e_value=float(e_value),
        e_value_ci=float(e_value_ci),
        interpretation=interpretation,
    )


def e_value_from_cohens_d(d: float, ci_lower_d: Optional[float] = None) -> EValueResult:
    """Compute E-value from Cohen's d (standardized mean difference).

    Converts d to an approximate risk ratio using the formula:
    RR ≈ exp(0.91 * d)
    """
    rr = np.exp(0.91 * abs(d))
    ci_lower_rr = np.exp(0.91 * abs(ci_lower_d)) if ci_lower_d is not None else None
    result = compute_e_value(rr, ci_lower_rr)
    result.metadata["cohens_d"] = d
    return result


# ============================================================================
# Rosenbaum Bounds
# ============================================================================

def rosenbaum_bounds(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    gamma_range: Optional[List[float]] = None,
    alpha: float = 0.05,
) -> RosenbaumResult:
    """Rosenbaum sensitivity analysis for matched pairs.

    Tests how sensitive a Wilcoxon signed-rank test result is to
    hidden bias in the matching. Gamma represents the maximum
    odds ratio of differential treatment assignment due to unobserved
    confounders.

    Args:
        treated_outcomes: Outcomes for treated units (matched pairs)
        control_outcomes: Outcomes for control units (matched pairs)
        gamma_range: Values of gamma to test (default: 1.0 to 5.0)
        alpha: Significance level

    Returns:
        RosenbaumResult with critical gamma
    """
    if gamma_range is None:
        gamma_range = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

    n = len(treated_outcomes)
    if n != len(control_outcomes):
        raise ValueError("Treated and control must have same length (matched pairs)")

    differences = treated_outcomes - control_outcomes
    abs_diff = np.abs(differences)
    signs = np.sign(differences)

    # Rank absolute differences
    nonzero_mask = abs_diff > 1e-12
    ranks = np.zeros(n)
    if nonzero_mask.sum() > 0:
        order = np.argsort(abs_diff[nonzero_mask])
        ranks[nonzero_mask] = np.arange(1, nonzero_mask.sum() + 1)[
            np.argsort(order)
        ]

    p_values = []
    for gamma in gamma_range:
        # Under hidden bias gamma, probability of correct treatment assignment
        p_plus = gamma / (1 + gamma)
        p_minus = 1 / (1 + gamma)

        # Expected value and variance of test statistic under Gamma-bias
        W_plus = np.sum(ranks[signs > 0])
        E_W = np.sum(ranks * p_plus)
        Var_W = np.sum(ranks ** 2 * p_plus * p_minus)

        if Var_W > 0:
            z = (W_plus - E_W) / np.sqrt(Var_W)
            # One-sided p-value
            from scipy.stats import norm
            p = 1 - norm.cdf(z)
        else:
            p = 0.5

        p_values.append(float(p))

    # Find critical gamma (smallest gamma where p > alpha)
    critical_gamma = gamma_range[-1]
    for g, p in zip(gamma_range, p_values):
        if p > alpha:
            critical_gamma = g
            break

    if critical_gamma > 3.0:
        robustness = "robust to substantial hidden bias"
    elif critical_gamma > 2.0:
        robustness = "moderately sensitive"
    elif critical_gamma > 1.5:
        robustness = "sensitive to hidden bias"
    else:
        robustness = "very sensitive — result may be due to confounding"

    interpretation = (
        f"Critical Γ = {critical_gamma:.2f}: the result remains significant "
        f"(p < {alpha}) for hidden bias up to Γ = {critical_gamma:.2f}. "
        f"Finding is {robustness}."
    )

    return RosenbaumResult(
        gamma_values=gamma_range,
        p_values=p_values,
        critical_gamma=critical_gamma,
        interpretation=interpretation,
    )


# ============================================================================
# Manski Bounds
# ============================================================================

def manski_bounds(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    selection: np.ndarray,
    outcome_range: Optional[Tuple[float, float]] = None,
) -> ManskiBoundsResult:
    """Manski worst-case bounds under selection.

    When outcomes are only observed for a selected subpopulation
    (e.g., students who completed the program), Manski bounds give
    the worst-case treatment effect under arbitrary selection.

    Args:
        outcomes: Observed outcomes (may contain NaN for unobserved)
        treatment: Binary treatment indicator (0/1)
        selection: Binary selection indicator (1 = outcome observed)
        outcome_range: (min, max) possible outcome values

    Returns:
        ManskiBoundsResult with bounds on treatment effect
    """
    observed_mask = selection.astype(bool)

    if outcome_range is None:
        y_min = np.nanmin(outcomes[observed_mask])
        y_max = np.nanmax(outcomes[observed_mask])
    else:
        y_min, y_max = outcome_range

    treated = treatment.astype(bool)

    # Point estimate (among observed)
    y1_obs = outcomes[treated & observed_mask]
    y0_obs = outcomes[~treated & observed_mask]
    if len(y1_obs) == 0 or len(y0_obs) == 0:
        return ManskiBoundsResult(
            lower_bound=float("nan"),
            upper_bound=float("nan"),
            point_estimate=float("nan"),
            selection_fraction=0.0,
            interpretation="Insufficient data for bounds computation.",
        )

    point_est = float(np.mean(y1_obs) - np.mean(y0_obs))

    # Selection fractions
    p_sel_t = np.mean(observed_mask[treated])
    p_sel_c = np.mean(observed_mask[~treated])
    p_t = np.mean(treated)

    # Manski bounds
    # E[Y(1)] bounds
    e_y1_lower = p_sel_t * np.mean(y1_obs) + (1 - p_sel_t) * y_min
    e_y1_upper = p_sel_t * np.mean(y1_obs) + (1 - p_sel_t) * y_max

    # E[Y(0)] bounds
    e_y0_lower = p_sel_c * np.mean(y0_obs) + (1 - p_sel_c) * y_min
    e_y0_upper = p_sel_c * np.mean(y0_obs) + (1 - p_sel_c) * y_max

    lower = float(e_y1_lower - e_y0_upper)
    upper = float(e_y1_upper - e_y0_lower)

    sel_frac = float(1 - np.mean(observed_mask))

    if lower > 0:
        sign_info = "positive even in worst case"
    elif upper < 0:
        sign_info = "negative even in worst case"
    else:
        sign_info = "sign is ambiguous — bounds include zero"

    interpretation = (
        f"Manski bounds: [{lower:.4f}, {upper:.4f}]. "
        f"Point estimate: {point_est:.4f}. "
        f"Selection rate: {1-sel_frac:.1%} observed. "
        f"Treatment effect is {sign_info}."
    )

    return ManskiBoundsResult(
        lower_bound=lower,
        upper_bound=upper,
        point_estimate=point_est,
        selection_fraction=sel_frac,
        interpretation=interpretation,
    )
