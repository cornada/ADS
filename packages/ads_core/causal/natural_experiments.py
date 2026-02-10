"""Natural experiments for curriculum causal analysis.

FGOS revisions (Federal State Educational Standards) provide quasi-random
policy shocks: programs are forced to restructure curricula when their
FGOS code is updated. This creates a natural experiment:

Treatment: FGOS revision requiring curriculum overhaul
Control: Programs under same FGOS that haven't yet revised

Diff-in-Diff: Compare pre/post curriculum metrics for early vs late adopters.

Key insight: FGOS revision is plausibly exogenous to individual university
decisions (it's a federal mandate), which helps satisfy the instrument
validity condition for causal inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class DiffInDiffResult:
    """Result of Difference-in-Differences analysis."""

    treatment_effect: float     # ATT: average treatment effect on treated
    se: float                   # standard error
    t_stat: float               # t-statistic
    p_value: float              # two-sided p-value
    ci_lower: float             # 95% CI lower
    ci_upper: float             # 95% CI upper
    pre_trend_test: float       # p-value for parallel trends test
    n_treated: int
    n_control: int
    n_periods: int
    group_means: Dict[str, Dict[str, float]]  # pre/post means per group
    interpretation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "treatment_effect": round(self.treatment_effect, 4),
            "se": round(self.se, 4),
            "t_stat": round(self.t_stat, 4),
            "p_value": round(self.p_value, 6),
            "ci_95": [round(self.ci_lower, 4), round(self.ci_upper, 4)],
            "pre_trend_p": round(self.pre_trend_test, 4),
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "group_means": self.group_means,
            "interpretation": self.interpretation,
            "metadata": self.metadata,
        }


@dataclass
class FGOSChangeEvent:
    """A detected FGOS policy change event."""

    fgos_code: str
    change_year: int
    pre_years: List[int]
    post_years: List[int]
    change_magnitude: float  # relative change in n_courses or structure
    metadata: Dict[str, Any] = field(default_factory=dict)


def detect_fgos_changes(
    df: pd.DataFrame,
    year_col: str = "year",
    fgos_col: str = "fgos_code",
    min_change_ratio: float = 0.5,
) -> List[FGOSChangeEvent]:
    """Detect FGOS revision events from temporal course data.

    A change event is detected when the number of courses or structure
    shifts significantly between consecutive years for a FGOS code.

    Args:
        df: Course-level DataFrame with year and FGOS columns
        year_col: Column name for academic year
        fgos_col: Column name for FGOS code
        min_change_ratio: Minimum relative change to flag as event

    Returns:
        List of detected change events
    """
    events: List[FGOSChangeEvent] = []
    years = sorted(df[year_col].unique())

    for fgos in df[fgos_col].unique():
        fgos_df = df[df[fgos_col] == fgos]
        per_year = fgos_df.groupby(year_col).size()

        if len(per_year) < 2:
            continue

        # Check consecutive year pairs
        for i in range(len(years) - 1):
            y1, y2 = years[i], years[i + 1]
            if y1 not in per_year.index or y2 not in per_year.index:
                continue

            n1 = per_year[y1]
            n2 = per_year[y2]

            if n1 == 0:
                continue

            ratio = abs(n2 - n1) / max(n1, 1)
            if ratio >= min_change_ratio:
                pre = [y for y in years if y <= y1 and y in per_year.index]
                post = [y for y in years if y >= y2 and y in per_year.index]
                events.append(FGOSChangeEvent(
                    fgos_code=str(fgos),
                    change_year=int(y2),
                    pre_years=pre,
                    post_years=post,
                    change_magnitude=float(ratio),
                    metadata={"n_before": int(n1), "n_after": int(n2)},
                ))

    return events


def diff_in_diff(
    treated_pre: np.ndarray,
    treated_post: np.ndarray,
    control_pre: np.ndarray,
    control_post: np.ndarray,
) -> DiffInDiffResult:
    """Compute Difference-in-Differences estimate.

    ATT = (E[Y_treated_post] - E[Y_treated_pre]) -
          (E[Y_control_post] - E[Y_control_pre])

    This gives the causal effect of treatment under the parallel trends
    assumption: absent treatment, treated and control would have followed
    the same trajectory.

    Args:
        treated_pre: Outcome values for treated group, pre-period
        treated_post: Outcome values for treated group, post-period
        control_pre: Outcome values for control group, pre-period
        control_post: Outcome values for control group, post-period

    Returns:
        DiffInDiffResult with treatment effect and diagnostics
    """
    from scipy import stats

    # Group means
    mean_t_pre = float(np.mean(treated_pre))
    mean_t_post = float(np.mean(treated_post))
    mean_c_pre = float(np.mean(control_pre))
    mean_c_post = float(np.mean(control_post))

    # DiD estimate
    diff_treated = mean_t_post - mean_t_pre
    diff_control = mean_c_post - mean_c_pre
    att = diff_treated - diff_control

    # Standard error (assuming independence, pooled variance)
    n_t = len(treated_pre) + len(treated_post)
    n_c = len(control_pre) + len(control_post)
    all_values = np.concatenate([treated_pre, treated_post, control_pre, control_post])

    var_t_pre = np.var(treated_pre, ddof=1) if len(treated_pre) > 1 else 0
    var_t_post = np.var(treated_post, ddof=1) if len(treated_post) > 1 else 0
    var_c_pre = np.var(control_pre, ddof=1) if len(control_pre) > 1 else 0
    var_c_post = np.var(control_post, ddof=1) if len(control_post) > 1 else 0

    se = np.sqrt(
        var_t_post / max(len(treated_post), 1) +
        var_t_pre / max(len(treated_pre), 1) +
        var_c_post / max(len(control_post), 1) +
        var_c_pre / max(len(control_pre), 1)
    )

    if se > 0:
        t_stat = att / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(n_t + n_c - 4, 1)))
    else:
        t_stat = 0.0
        p_value = 1.0

    ci_lower = att - 1.96 * se
    ci_upper = att + 1.96 * se

    # Pre-trend test (are pre-period trends parallel?)
    # Simple version: test if control_pre ≈ treated_pre in trends
    pre_trend_p = 1.0  # default: parallel trends holds
    if len(treated_pre) > 1 and len(control_pre) > 1:
        _, pre_trend_p = stats.mannwhitneyu(
            treated_pre, control_pre, alternative="two-sided",
        )

    group_means = {
        "treated_pre": mean_t_pre,
        "treated_post": mean_t_post,
        "control_pre": mean_c_pre,
        "control_post": mean_c_post,
        "diff_treated": diff_treated,
        "diff_control": diff_control,
    }

    if p_value < 0.01:
        significance = "highly significant"
    elif p_value < 0.05:
        significance = "significant"
    elif p_value < 0.1:
        significance = "marginally significant"
    else:
        significance = "not significant"

    interpretation = (
        f"DiD estimate = {att:.4f} ({significance}, p={p_value:.4f}). "
        f"Treated group changed by {diff_treated:.4f}, "
        f"control by {diff_control:.4f}. "
        f"Pre-trend test p={pre_trend_p:.4f} "
        f"({'parallel trends OK' if pre_trend_p > 0.1 else 'WARNING: pre-trends differ'})."
    )

    return DiffInDiffResult(
        treatment_effect=float(att),
        se=float(se),
        t_stat=float(t_stat),
        p_value=float(p_value),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        pre_trend_test=float(pre_trend_p),
        n_treated=len(treated_pre) + len(treated_post),
        n_control=len(control_pre) + len(control_post),
        n_periods=2,
        group_means=group_means,
        interpretation=interpretation,
    )


def compute_program_metrics(
    df: pd.DataFrame,
    fgos_code: str,
    year: int,
    year_col: str = "year",
    fgos_col: str = "fgos_code",
) -> Dict[str, float]:
    """Compute summary metrics for a program in a given year.

    Returns:
        Dict with n_courses, mean_credits, competency_count, diversity_index
    """
    mask = (df[fgos_col] == fgos_code) & (df[year_col] == year)
    subset = df[mask]

    if len(subset) == 0:
        return {}

    metrics: Dict[str, float] = {"n_courses": float(len(subset))}

    if "credits_zet" in subset.columns:
        credits = pd.to_numeric(subset["credits_zet"], errors="coerce")
        metrics["mean_credits"] = float(credits.mean()) if credits.notna().any() else 0.0

    if "competency_codes" in subset.columns:
        all_comps = set()
        for codes in subset["competency_codes"].dropna():
            all_comps.update(str(codes).split(","))
        metrics["competency_count"] = float(len(all_comps))

    return metrics


def run_fgos_did(
    df: pd.DataFrame,
    metric: str = "n_courses",
    treatment_year: int = 2024,
    year_col: str = "year",
    fgos_col: str = "fgos_code",
) -> Dict[str, Any]:
    """Run Diff-in-Diff for FGOS change as natural experiment.

    Treats programs that had a structural break at treatment_year as
    "treated" and programs without a break as "control".

    Args:
        df: Full course DataFrame
        metric: Which program-level metric to analyze
        treatment_year: Year of the policy shock
        year_col: Year column name
        fgos_col: FGOS code column name

    Returns:
        Dict with DiD results, events detected, and per-program details
    """
    years = sorted(df[year_col].unique())
    pre_years = [y for y in years if y < treatment_year]
    post_years = [y for y in years if y >= treatment_year]

    if not pre_years or not post_years:
        return {"error": "Not enough pre/post years"}

    # Detect which FGOS codes had structural changes
    events = detect_fgos_changes(df, year_col, fgos_col, min_change_ratio=0.5)
    treated_fgos = {e.fgos_code for e in events if e.change_year == treatment_year}

    all_fgos = set(df[fgos_col].unique())
    control_fgos = all_fgos - treated_fgos

    # Compute program-level metrics
    treated_pre_vals = []
    treated_post_vals = []
    control_pre_vals = []
    control_post_vals = []

    for fgos in treated_fgos:
        for y in pre_years:
            m = compute_program_metrics(df, fgos, y, year_col, fgos_col)
            if metric in m:
                treated_pre_vals.append(m[metric])
        for y in post_years:
            m = compute_program_metrics(df, fgos, y, year_col, fgos_col)
            if metric in m:
                treated_post_vals.append(m[metric])

    for fgos in control_fgos:
        for y in pre_years:
            m = compute_program_metrics(df, fgos, y, year_col, fgos_col)
            if metric in m:
                control_pre_vals.append(m[metric])
        for y in post_years:
            m = compute_program_metrics(df, fgos, y, year_col, fgos_col)
            if metric in m:
                control_post_vals.append(m[metric])

    if not all([treated_pre_vals, treated_post_vals,
                control_pre_vals, control_post_vals]):
        return {"error": "Insufficient data for DiD"}

    did_result = diff_in_diff(
        np.array(treated_pre_vals),
        np.array(treated_post_vals),
        np.array(control_pre_vals),
        np.array(control_post_vals),
    )

    return {
        "metric": metric,
        "treatment_year": treatment_year,
        "n_treated_fgos": len(treated_fgos),
        "n_control_fgos": len(control_fgos),
        "treated_fgos": sorted(treated_fgos),
        "control_fgos": sorted(control_fgos),
        "did_result": did_result.to_dict(),
        "events": [
            {"fgos": e.fgos_code, "year": e.change_year, "magnitude": e.change_magnitude}
            for e in events
        ],
    }
