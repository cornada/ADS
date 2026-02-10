"""Temporal evolution API endpoints.

Provides endpoints for analyzing how curricula change over time,
using changepoint detection and drift metrics.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import numpy as np

router = APIRouter()


# ============================================================================
# Schemas
# ============================================================================

class ChangePointResult(BaseModel):
    """Detected changepoint in a time series."""
    position: int
    year: Optional[str] = None
    metric: str
    magnitude: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DriftAnalysis(BaseModel):
    """Curriculum drift analysis between two time periods."""
    period_start: str
    period_end: str
    embedding_drift: float     # cosine distance between period centroids
    vocabulary_overlap: float  # Jaccard similarity of course names/topics
    structural_change: float   # fraction of courses added/removed
    n_courses_start: int
    n_courses_end: int


class TemporalOverview(BaseModel):
    """Overview of temporal data available."""
    years_available: List[str]
    n_courses_by_year: Dict[str, int]
    total_changepoints: int
    top_drift_periods: List[DriftAnalysis]


# ============================================================================
# In-memory data (loaded at startup)
# ============================================================================

_temporal_data: Dict[str, Any] = {}


def load_temporal_data(data: Dict[str, Any]):
    """Load temporal analysis results."""
    global _temporal_data
    _temporal_data = data


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/overview", response_model=TemporalOverview)
def temporal_overview():
    """Get overview of temporal data and detected changes."""
    if not _temporal_data:
        return TemporalOverview(
            years_available=[],
            n_courses_by_year={},
            total_changepoints=0,
            top_drift_periods=[],
        )

    return TemporalOverview(**_temporal_data.get("overview", {
        "years_available": [],
        "n_courses_by_year": {},
        "total_changepoints": 0,
        "top_drift_periods": [],
    }))


@router.get("/changepoints")
def list_changepoints(
    metric: Optional[str] = None,
    min_magnitude: float = 0.0,
):
    """List detected changepoints with optional filtering."""
    changepoints = _temporal_data.get("changepoints", [])

    if metric:
        changepoints = [cp for cp in changepoints if cp.get("metric") == metric]
    if min_magnitude > 0:
        changepoints = [cp for cp in changepoints if cp.get("magnitude", 0) >= min_magnitude]

    return {"changepoints": changepoints, "count": len(changepoints)}


@router.get("/drift/{year_a}/{year_b}")
def compare_years(year_a: str, year_b: str):
    """Compare curricula between two years."""
    drift_data = _temporal_data.get("drift_pairs", {})
    key = f"{year_a}_{year_b}"

    if key not in drift_data:
        return {
            "period_start": year_a,
            "period_end": year_b,
            "message": "No drift data available for this pair. Available pairs: " +
                       ", ".join(drift_data.keys()),
        }

    return drift_data[key]


@router.get("/evolution/{program_code}")
def program_evolution(program_code: str):
    """Track a specific program's evolution over time."""
    evolutions = _temporal_data.get("program_evolution", {})

    if program_code not in evolutions:
        available = list(evolutions.keys())[:10]
        raise HTTPException(
            status_code=404,
            detail=f"Program not found: {program_code}. Available: {available}",
        )

    return evolutions[program_code]
