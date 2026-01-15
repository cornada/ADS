"""Analysis modules for ADS experiments.

Provides:
- Outcome sanity checking (predicted vs observed)
- Correlation and calibration metrics
- Baseline comparisons
"""
from ads_core.analysis.outcome_sanity import (
    OutcomeSanityResult,
    compute_outcome_sanity,
    aggregate_sanity_results,
)

__all__ = [
    "OutcomeSanityResult",
    "compute_outcome_sanity",
    "aggregate_sanity_results",
]
