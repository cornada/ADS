"""Outcomes normalization and analysis for ADS.

Provides:
- Canonical outcomes schema
- Normalization from UCB/ASU formats
- Provenance tracking
"""
from ads_core.outcomes.normalize import (
    CanonicalOutcome,
    normalize_ucb_outcomes,
    normalize_asu_outcomes,
    load_normalized_outcomes,
)

__all__ = [
    "CanonicalOutcome",
    "normalize_ucb_outcomes",
    "normalize_asu_outcomes",
    "load_normalized_outcomes",
]
