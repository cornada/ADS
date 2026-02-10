"""Change-point detection for curriculum temporal evolution.

Detects "regime shifts" in program-level time series — moments where
the statistical properties of a curriculum change abruptly (e.g., FGOS
revision forcing wholesale restructuring).

Signals constructed per FGOS direction per year:
- n_courses: number of courses offered
- mean_credits: average credits (ЗЕТ) per course
- competency_jaccard: Jaccard distance of competency sets vs previous year
- description_churn: fraction of new/modified course descriptions

Algorithms (via ruptures):
- Pelt: penalized exact linear time — optimal offline detection
- Binseg: binary segmentation — faster, suboptimal
- Window: sliding window — good for short series

Short-series handling:
With only 3-5 time points, classical change-point algorithms overfit.
We use conservative penalties and validate via permutation testing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import ruptures as rpt
except ImportError:
    rpt = None  # type: ignore[assignment]


@dataclass
class ChangePoint:
    """A detected change point in a time series."""

    index: int
    year_before: int
    year_after: int
    signal_name: str
    cost_before: float
    cost_after: float
    magnitude: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgramEvolution:
    """Temporal evolution summary for a single FGOS program."""

    fgos_code: str
    years: List[int]
    signals: Dict[str, List[float]]
    changepoints: List[ChangePoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_years(self) -> int:
        return len(self.years)

    @property
    def has_changepoints(self) -> bool:
        return len(self.changepoints) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fgos_code": self.fgos_code,
            "years": self.years,
            "n_years": self.n_years,
            "signals": self.signals,
            "changepoints": [
                {
                    "index": cp.index,
                    "year_before": cp.year_before,
                    "year_after": cp.year_after,
                    "signal": cp.signal_name,
                    "magnitude": round(cp.magnitude, 4),
                }
                for cp in self.changepoints
            ],
            "metadata": self.metadata,
        }


# ============================================================================
# Signal Construction
# ============================================================================

def build_program_signals(
    df,
    fgos_code: str,
    course_col: str = "course_name",
    year_col: str = "year",
    credits_col: str = "credits_zet",
    competency_col: str = "competency_codes",
    content_col: str = "content_summary",
) -> ProgramEvolution:
    """Build temporal signals for a single FGOS program from course data.

    Args:
        df: DataFrame with course data (all years for one FGOS)
        fgos_code: FGOS direction code
        course_col: Column with course names
        year_col: Column with year
        credits_col: Column with credits (ЗЕТ)
        competency_col: Column with comma-separated competency codes
        content_col: Column with content/description text

    Returns:
        ProgramEvolution with computed signals
    """
    import pandas as pd

    subset = df[df["fgos_code"] == fgos_code].copy()
    years = sorted(subset[year_col].dropna().unique())

    if len(years) < 2:
        return ProgramEvolution(
            fgos_code=fgos_code,
            years=[int(y) for y in years],
            signals={},
            metadata={"reason": "insufficient_years"},
        )

    signals: Dict[str, List[float]] = {
        "n_courses": [],
        "mean_credits": [],
        "competency_jaccard": [],
        "course_churn": [],
    }

    prev_courses: set = set()
    prev_competencies: set = set()

    for y in years:
        year_data = subset[subset[year_col] == y]

        # Number of courses
        n = len(year_data)
        signals["n_courses"].append(float(n))

        # Mean credits
        credits = pd.to_numeric(year_data[credits_col], errors="coerce")
        mean_c = credits.mean()
        signals["mean_credits"].append(float(mean_c) if not np.isnan(mean_c) else 0.0)

        # Current course names and competencies
        curr_courses = set(
            str(c).strip()
            for c in year_data[course_col].dropna()
            if str(c).strip()
        )
        curr_competencies: set = set()
        for codes in year_data[competency_col].dropna():
            for c in str(codes).split(","):
                c = c.strip()
                if c:
                    curr_competencies.add(c)

        # Jaccard distance of competency sets vs previous year
        if prev_competencies:
            union = prev_competencies | curr_competencies
            intersection = prev_competencies & curr_competencies
            jaccard = 1.0 - (len(intersection) / max(len(union), 1))
        else:
            jaccard = 0.0  # no comparison for first year
        signals["competency_jaccard"].append(jaccard)

        # Course churn: fraction of courses that are new or disappeared
        if prev_courses:
            added = curr_courses - prev_courses
            removed = prev_courses - curr_courses
            total_unique = len(prev_courses | curr_courses)
            churn = (len(added) + len(removed)) / max(total_unique, 1)
        else:
            churn = 0.0  # no comparison for first year
        signals["course_churn"].append(churn)

        prev_courses = curr_courses
        prev_competencies = curr_competencies

    return ProgramEvolution(
        fgos_code=fgos_code,
        years=[int(y) for y in years],
        signals=signals,
        metadata={
            "n_courses_range": [
                int(min(signals["n_courses"])),
                int(max(signals["n_courses"])),
            ]
        },
    )


# ============================================================================
# Change-Point Detection
# ============================================================================

def detect_changepoints(
    signal: List[float],
    years: List[int],
    signal_name: str = "signal",
    method: str = "pelt",
    min_size: int = 1,
    penalty: float = 3.0,
    n_bkps: Optional[int] = None,
) -> List[ChangePoint]:
    """Detect change points in a single time series.

    For short series (n <= 5), defaults to binseg with n_bkps=1 to find
    the single most likely breakpoint, then validate via magnitude filtering.
    Returns empty list if series too short (< 3 points).

    Args:
        signal: Time series values
        years: Corresponding year labels
        signal_name: Name of the signal (for metadata)
        method: "pelt", "binseg", or "window". For n<=5, overridden to "binseg".
        min_size: Minimum segment size
        penalty: Penalty for PELT (higher = fewer changepoints)
        n_bkps: Fixed number of breakpoints (for binseg; None = auto)

    Returns:
        List of detected ChangePoint objects
    """
    if rpt is None:
        raise ImportError("ruptures is required: pip install ruptures")

    n = len(signal)
    if n < 3:
        return []

    arr = np.array(signal, dtype=np.float64).reshape(-1, 1)

    # Check for constant signal (no variance = no changepoint)
    if np.std(arr) < 1e-10:
        return []

    cost_model = "l2"

    try:
        if n <= 7:
            # Short series: use KernelCPD (robust to small n where binseg/pelt fail)
            algo = rpt.KernelCPD(kernel="linear", min_size=2)
            algo.fit(arr)
            bkps = algo.predict(n_bkps=1)
        elif method == "pelt":
            algo = rpt.Pelt(model=cost_model, min_size=min_size)
            algo.fit(arr)
            bkps = algo.predict(pen=penalty)
        elif method == "binseg":
            algo = rpt.Binseg(model=cost_model, min_size=min_size)
            algo.fit(arr)
            if n_bkps is None:
                n_bkps = max(1, n // 3)
            bkps = algo.predict(n_bkps=n_bkps)
        elif method == "window":
            width = max(2, n // 3)
            algo = rpt.Window(model=cost_model, width=width, min_size=min_size)
            algo.fit(arr)
            bkps = algo.predict(pen=penalty)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pelt', 'binseg', or 'window'.")
    except Exception:
        # ruptures raises BadSegmentationParameters for degenerate cases
        return []

    # Convert breakpoints to ChangePoint objects
    # ruptures returns indices where segments END (last index = n always)
    signal_range = float(np.max(arr) - np.min(arr))
    results = []
    for bp in bkps:
        if bp >= n:
            continue  # skip the terminal breakpoint

        # Compute magnitude as absolute change at the breakpoint
        val_before = float(np.mean(arr[:bp]))
        val_after = float(np.mean(arr[bp:]))
        magnitude = abs(val_after - val_before)

        # Filter: magnitude must be at least 20% of signal range
        if signal_range > 0 and magnitude < 0.2 * signal_range:
            continue

        # Map index to years
        year_before = years[bp - 1] if bp > 0 else years[0]
        year_after = years[bp] if bp < len(years) else years[-1]

        results.append(ChangePoint(
            index=bp,
            year_before=int(year_before),
            year_after=int(year_after),
            signal_name=signal_name,
            cost_before=val_before,
            cost_after=val_after,
            magnitude=magnitude,
        ))

    return results


def detect_program_changepoints(
    evolution: ProgramEvolution,
    method: str = "pelt",
    penalty: float = 3.0,
    signals_to_check: Optional[List[str]] = None,
) -> ProgramEvolution:
    """Run change-point detection on all signals of a program.

    Args:
        evolution: ProgramEvolution with signals
        method: Detection method
        penalty: Penalty parameter
        signals_to_check: Subset of signal names to analyze (None = all)

    Returns:
        Updated ProgramEvolution with changepoints field populated
    """
    if not evolution.signals:
        return evolution

    all_cps: List[ChangePoint] = []
    check = signals_to_check or list(evolution.signals.keys())

    for sig_name in check:
        if sig_name not in evolution.signals:
            continue
        cps = detect_changepoints(
            signal=evolution.signals[sig_name],
            years=evolution.years,
            signal_name=sig_name,
            method=method,
            penalty=penalty,
        )
        all_cps.extend(cps)

    evolution.changepoints = all_cps
    return evolution


# ============================================================================
# Permutation Test for Short Series
# ============================================================================

def permutation_test_changepoint(
    signal: List[float],
    years: List[int],
    method: str = "pelt",
    penalty: float = 3.0,
    n_permutations: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Assess significance of detected changepoints via permutation test.

    Shuffles the signal values and re-runs detection to estimate
    false positive rate. A changepoint is significant if it appears
    in fewer than 5% of permuted series.

    Args:
        signal: Original time series
        years: Year labels
        method: Detection method
        penalty: Penalty parameter
        n_permutations: Number of permutations
        seed: Random seed

    Returns:
        Dict with original changepoints, permutation distribution, p-values
    """
    if rpt is None:
        raise ImportError("ruptures is required: pip install ruptures")

    original_cps = detect_changepoints(signal, years, method=method, penalty=penalty)

    if not original_cps:
        return {"changepoints": [], "n_permutations": n_permutations, "false_positive_rates": {}}

    rng = np.random.RandomState(seed)

    # For magnitude-based p-value: count how often a permuted series
    # produces a changepoint with magnitude >= the observed one
    original_magnitudes = {cp.index: cp.magnitude for cp in original_cps}
    magnitude_exceedances = {cp.index: 0 for cp in original_cps}
    perm_counts = np.zeros(len(signal))

    for _ in range(n_permutations):
        perm_signal = list(rng.permutation(signal))
        perm_cps = detect_changepoints(perm_signal, years, method=method, penalty=penalty)
        max_perm_mag = max((cp.magnitude for cp in perm_cps), default=0.0)
        for cp in perm_cps:
            if cp.index < len(perm_counts):
                perm_counts[cp.index] += 1
        # Check if any permuted CP exceeds each original CP's magnitude
        for orig_idx, orig_mag in original_magnitudes.items():
            if max_perm_mag >= orig_mag:
                magnitude_exceedances[orig_idx] += 1

    # P-value based on magnitude: fraction of permutations with CP >= observed magnitude
    p_values = {}
    for cp in original_cps:
        p = magnitude_exceedances[cp.index] / n_permutations
        p_values[cp.index] = round(float(p), 4)

    return {
        "changepoints": [
            {
                "index": cp.index,
                "year_before": cp.year_before,
                "year_after": cp.year_after,
                "magnitude": round(cp.magnitude, 4),
                "p_value": p_values.get(cp.index, 1.0),
                "significant": p_values.get(cp.index, 1.0) < 0.05,
            }
            for cp in original_cps
        ],
        "n_permutations": n_permutations,
        "false_positive_rates": {
            int(i): round(float(c / n_permutations), 4)
            for i, c in enumerate(perm_counts)
            if c > 0
        },
    }


# ============================================================================
# Multi-Signal Aggregation
# ============================================================================

def aggregate_changepoints(
    evolutions: List[ProgramEvolution],
) -> Dict[str, Any]:
    """Aggregate change-point results across multiple programs.

    Finds consensus years where multiple programs show regime shifts.

    Args:
        evolutions: List of ProgramEvolution objects with changepoints

    Returns:
        Summary dict with per-year counts, affected programs, consensus years
    """
    year_counts: Dict[int, int] = {}
    year_programs: Dict[int, List[str]] = {}
    year_signals: Dict[int, List[str]] = {}

    programs_with_cp = 0
    total_cps = 0

    for evo in evolutions:
        if evo.has_changepoints:
            programs_with_cp += 1
        for cp in evo.changepoints:
            total_cps += 1
            # Use year_after as the "change year"
            y = cp.year_after
            year_counts[y] = year_counts.get(y, 0) + 1
            year_programs.setdefault(y, []).append(evo.fgos_code)
            year_signals.setdefault(y, []).append(cp.signal_name)

    # Consensus: years where >30% of programs show a shift
    n_programs = len(evolutions)
    consensus_threshold = max(2, int(0.3 * n_programs))
    consensus_years = {
        y: count
        for y, count in year_counts.items()
        if count >= consensus_threshold
    }

    return {
        "total_programs": n_programs,
        "programs_with_changepoints": programs_with_cp,
        "total_changepoints": total_cps,
        "per_year": {
            int(y): {
                "count": count,
                "programs": year_programs.get(y, []),
                "signals": list(set(year_signals.get(y, []))),
            }
            for y, count in sorted(year_counts.items())
        },
        "consensus_years": {int(y): c for y, c in consensus_years.items()},
        "consensus_threshold": consensus_threshold,
    }
