"""Explainability and recourse module for ADS recommendations.

Provides:
- Evidence-based explanations for why options score well on objectives
- Recourse suggestions for improving specific objectives
- Counterfactual analysis for Pareto membership
"""
from ads_core.explain.explainer import (
    Explanation,
    RecourseOption,
    Explainer,
    explain_option,
    suggest_recourse,
)

__all__ = [
    "Explanation",
    "RecourseOption",
    "Explainer",
    "explain_option",
    "suggest_recourse",
]
