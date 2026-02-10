"""Optimal Transport API endpoints.

Provides endpoints for computing OT distances between
curriculum and market embedding distributions.
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

class OTDistanceRequest(BaseModel):
    """Request to compute OT distance between two distributions."""
    source_ids: List[str] = Field(..., description="Course/artifact IDs for source")
    target_ids: List[str] = Field(..., description="Course/artifact IDs for target")
    metric: str = Field(default="cosine", description="Distance metric: cosine | euclidean")


class OTDistanceResult(BaseModel):
    """OT distance between two distributions."""
    wasserstein_distance: float
    n_source: int
    n_target: int
    metric: str
    transport_plan_summary: Dict[str, Any] = Field(default_factory=dict)


class MarketAlignmentResult(BaseModel):
    """Market alignment for a university/program."""
    university: str
    program: str = ""
    market_source: str
    ot_distance: float
    n_courses: int
    n_market_items: int
    top_aligned_courses: List[Dict[str, Any]] = Field(default_factory=list)
    top_gaps: List[Dict[str, Any]] = Field(default_factory=list)


# ============================================================================
# In-memory data
# ============================================================================

_embeddings: Dict[str, np.ndarray] = {}
_alignment_cache: Dict[str, MarketAlignmentResult] = {}


def load_embeddings(embeddings: Dict[str, np.ndarray]):
    """Load embeddings for OT computation."""
    global _embeddings
    _embeddings = embeddings


def load_alignment_cache(cache: Dict[str, MarketAlignmentResult]):
    """Load pre-computed alignment results."""
    global _alignment_cache
    _alignment_cache = cache


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/distance", response_model=OTDistanceResult)
def compute_ot_distance(req: OTDistanceRequest):
    """Compute Wasserstein distance between two sets of embeddings."""
    source_embs = [_embeddings[sid] for sid in req.source_ids if sid in _embeddings]
    target_embs = [_embeddings[tid] for tid in req.target_ids if tid in _embeddings]

    if len(source_embs) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 source embeddings")
    if len(target_embs) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 target embeddings")

    X = np.array(source_embs)
    Y = np.array(target_embs)

    # Compute cost matrix
    if req.metric == "cosine":
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
        cost = 1 - X_norm @ Y_norm.T
    else:
        from scipy.spatial.distance import cdist
        cost = cdist(X, Y, metric="euclidean")

    # Uniform weights
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)

    try:
        import ot
        T = ot.emd(a, b, cost)
        distance = float(np.sum(T * cost))
    except ImportError:
        # Fallback: approximate with mean pairwise distance
        distance = float(np.mean(cost))
        T = None

    summary = {"method": "emd" if T is not None else "mean_approx"}
    if T is not None:
        # Top transport pairs
        flat_indices = np.argsort(T.ravel())[::-1][:5]
        top_pairs = []
        for idx in flat_indices:
            i, j = divmod(idx, len(Y))
            if T[i, j] > 1e-6:
                top_pairs.append({
                    "source_idx": int(i),
                    "target_idx": int(j),
                    "mass": float(T[i, j]),
                    "cost": float(cost[i, j]),
                })
        summary["top_transport_pairs"] = top_pairs

    return OTDistanceResult(
        wasserstein_distance=distance,
        n_source=len(X),
        n_target=len(Y),
        metric=req.metric,
        transport_plan_summary=summary,
    )


@router.get("/alignment/{university}")
def get_market_alignment(university: str, market: str = "headhunter"):
    """Get pre-computed market alignment for a university."""
    key = f"{university.lower()}_{market.lower()}"

    if key not in _alignment_cache:
        available = list(_alignment_cache.keys())
        raise HTTPException(
            status_code=404,
            detail=f"No alignment data for key '{key}'. Available: {available[:10]}",
        )

    return _alignment_cache[key]


@router.get("/alignment/compare")
def compare_alignments(
    universities: str = "MISIS,KTH",
    market: str = "headhunter",
):
    """Compare market alignment across universities."""
    uni_list = [u.strip() for u in universities.split(",")]
    results = {}

    for uni in uni_list:
        key = f"{uni.lower()}_{market.lower()}"
        if key in _alignment_cache:
            results[uni] = _alignment_cache[key]

    return {
        "market": market,
        "universities": list(results.keys()),
        "alignments": results,
    }
