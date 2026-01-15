"""Dashboard API endpoints for KT8.

Provides endpoints for:
- Getting Pareto data with full option details
- Dynamic recalculation with toggle controls
- Explanation and recourse for selected options
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

import numpy as np

from ads_core.data.schemas import ArtifactType
from ads_core.ingest.toy_dataset import build_toy_artifacts
from ads_core.embed.encoder import StubEncoder
from ads_core.embed.sentence_transformers_encoder import SentenceTransformersEncoder
from ads_core.lenses.identity import IdentityLens
from ads_core.lenses.diagonal import DiagonalLens
from ads_core.eval.objectives import centroid, evaluate_option
from ads_core.eval.pareto import pareto_front
from ads_core.eval.constraints import apply_constraints, autonomy_drift_detailed
from ads_core.explain.explainer import Explainer

router = APIRouter()


# Cache for embeddings (avoid recomputing on each request)
_embedding_cache: Dict[str, Any] = {}


def _get_or_compute_embeddings(embedding_kind: str = "stub", model_name: str = None):
    """Get or compute embeddings with caching."""
    cache_key = f"{embedding_kind}:{model_name or 'default'}"

    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    # Build toy dataset
    artifacts = build_toy_artifacts()

    # Select encoder
    if embedding_kind == "stub":
        enc = StubEncoder(d=64)
    elif embedding_kind == "sbert":
        enc = SentenceTransformersEncoder(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        raise ValueError(f"Unknown embedding kind: {embedding_kind}")

    # Compute embeddings
    texts = [a.text for a in artifacts]
    vecs = enc.encode(texts)
    id2vec = {artifacts[i].artifact_id: vecs[i] for i in range(len(artifacts))}

    result = {
        "artifacts": artifacts,
        "id2vec": id2vec,
        "encoder_id": enc.model_id,
    }
    _embedding_cache[cache_key] = result
    return result


class DashboardConfig(BaseModel):
    """Configuration for dashboard view."""

    objectives: List[str] = Field(
        default=["market", "mission", "university", "learner"],
        description="Enabled objectives"
    )
    autonomy_tau: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Autonomy drift threshold"
    )
    lens_mode: str = Field(
        default="identity",
        description="Lens mode: identity, diagonal"
    )
    embedding_kind: str = Field(
        default="stub",
        description="Embedding type: stub, sbert"
    )
    topk: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of options to evaluate"
    )


class OptionSummary(BaseModel):
    """Summary of an option for the dashboard."""

    option_id: str
    text: str
    objectives: Dict[str, float]
    feasible: bool
    is_pareto: bool
    autonomy_drift: float


class DashboardResponse(BaseModel):
    """Response with full dashboard data."""

    config: DashboardConfig
    options: List[OptionSummary]
    pareto_ids: List[str]
    objective_keys: List[str]
    stats: Dict[str, Any]


@router.post("/data", response_model=DashboardResponse)
def get_dashboard_data(config: DashboardConfig):
    """Get dashboard data with current configuration.

    This endpoint computes Pareto front and option scores based on
    the provided configuration (objectives, tau, lenses).
    """
    # Get embeddings
    data = _get_or_compute_embeddings(config.embedding_kind)
    artifacts = data["artifacts"]
    id2vec = data["id2vec"]

    # Build target centroids
    target_centroids = {}
    if "market" in config.objectives:
        market_vs = [id2vec[a.artifact_id] for a in artifacts if a.type == ArtifactType.JOB_ROLE]
        target_centroids["market"] = centroid(market_vs)
    if "mission" in config.objectives:
        mission_vs = [id2vec[a.artifact_id] for a in artifacts if a.type == ArtifactType.MISSION]
        target_centroids["mission"] = centroid(mission_vs)
    if "university" in config.objectives:
        univ_vs = [id2vec[a.artifact_id] for a in artifacts if a.type == ArtifactType.COURSE]
        target_centroids["university"] = centroid(univ_vs)
    if "learner" in config.objectives:
        learner_v = [id2vec[a.artifact_id] for a in artifacts if a.type == ArtifactType.LEARNER_PROFILE][0]
        target_centroids["learner"] = learner_v / (np.linalg.norm(learner_v) + 1e-8)

    # Build lenses
    if config.lens_mode == "identity":
        lenses = {k: IdentityLens(lens_id=f"identity:{k}") for k in target_centroids}
    elif config.lens_mode == "diagonal":
        d = next(iter(id2vec.values())).shape[0]
        w = np.ones((d,), dtype=np.float32)
        lenses = {k: DiagonalLens(lens_id=f"diag:{k}", weights=w) for k in target_centroids}
    else:
        lenses = {k: IdentityLens(lens_id=f"identity:{k}") for k in target_centroids}

    # Get courses
    courses = [a for a in artifacts if a.type == ArtifactType.COURSE]

    # Evaluate all courses
    results = []
    for c in courses[:config.topk]:
        obj = evaluate_option(id2vec[c.artifact_id], target_centroids, lenses)
        cons = apply_constraints(obj, tau=config.autonomy_tau)
        drift = autonomy_drift_detailed(obj)
        results.append({
            "option_id": c.artifact_id,
            "text": c.text,
            "objectives": obj,
            "feasible": cons["feasible"],
            "autonomy_drift": drift["total"],
        })

    # Compute Pareto front
    keys = list(target_centroids.keys())
    pareto_idx = pareto_front([r["objectives"] for r in results], keys=keys)
    pareto_ids = [results[i]["option_id"] for i in pareto_idx]

    # Build response
    options = []
    for r in results:
        options.append(OptionSummary(
            option_id=r["option_id"],
            text=r["text"],
            objectives={k: round(v, 4) for k, v in r["objectives"].items()},
            feasible=r["feasible"],
            is_pareto=r["option_id"] in pareto_ids,
            autonomy_drift=round(r["autonomy_drift"], 4),
        ))

    # Compute stats
    feasible_count = sum(1 for o in options if o.feasible)
    stats = {
        "total_options": len(options),
        "pareto_count": len(pareto_ids),
        "feasible_count": feasible_count,
        "infeasible_count": len(options) - feasible_count,
        "encoder": data["encoder_id"],
    }

    return DashboardResponse(
        config=config,
        options=options,
        pareto_ids=pareto_ids,
        objective_keys=keys,
        stats=stats,
    )


class ExplainRequest(BaseModel):
    """Request for explaining an option."""

    option_id: str
    config: DashboardConfig


class ExplainResponse(BaseModel):
    """Response with explanation data."""

    option_id: str
    explanation: Dict[str, Any]
    recourse: Optional[Dict[str, Any]] = None


@router.post("/explain", response_model=ExplainResponse)
def explain_option(req: ExplainRequest):
    """Get explanation for a specific option."""
    # Get embeddings and compute everything
    data = _get_or_compute_embeddings(req.config.embedding_kind)
    artifacts = data["artifacts"]
    id2vec = data["id2vec"]

    # Build targets
    target_centroids = {}
    target_artifacts = {}

    if "market" in req.config.objectives:
        market_arts = [a for a in artifacts if a.type == ArtifactType.JOB_ROLE]
        market_vs = [id2vec[a.artifact_id] for a in market_arts]
        target_centroids["market"] = centroid(market_vs)
        target_artifacts["market"] = [(a.artifact_id, a.text, id2vec[a.artifact_id]) for a in market_arts]

    if "mission" in req.config.objectives:
        mission_arts = [a for a in artifacts if a.type == ArtifactType.MISSION]
        mission_vs = [id2vec[a.artifact_id] for a in mission_arts]
        target_centroids["mission"] = centroid(mission_vs)
        target_artifacts["mission"] = [(a.artifact_id, a.text, id2vec[a.artifact_id]) for a in mission_arts]

    if "university" in req.config.objectives:
        univ_arts = [a for a in artifacts if a.type == ArtifactType.COURSE]
        univ_vs = [id2vec[a.artifact_id] for a in univ_arts]
        target_centroids["university"] = centroid(univ_vs)
        target_artifacts["university"] = [(a.artifact_id, a.text, id2vec[a.artifact_id]) for a in univ_arts]

    if "learner" in req.config.objectives:
        learner_arts = [a for a in artifacts if a.type == ArtifactType.LEARNER_PROFILE]
        learner_v = id2vec[learner_arts[0].artifact_id]
        target_centroids["learner"] = learner_v / (np.linalg.norm(learner_v) + 1e-8)
        target_artifacts["learner"] = [(a.artifact_id, a.text, id2vec[a.artifact_id]) for a in learner_arts]

    # Build lenses (matching the lens_mode from dashboard data)
    if req.config.lens_mode == "identity":
        lenses = {k: IdentityLens(lens_id=f"identity:{k}") for k in target_centroids}
    elif req.config.lens_mode == "diagonal":
        d = next(iter(id2vec.values())).shape[0]
        w = np.ones((d,), dtype=np.float32)
        lenses = {k: DiagonalLens(lens_id=f"diag:{k}", weights=w) for k in target_centroids}
    else:
        # Default to identity
        lenses = {k: IdentityLens(lens_id=f"identity:{k}") for k in target_centroids}

    # Get courses
    courses = [a for a in artifacts if a.type == ArtifactType.COURSE]
    course_vecs = {a.artifact_id: id2vec[a.artifact_id] for a in courses}
    course_texts = {a.artifact_id: a.text for a in courses}

    # Check if option exists
    if req.option_id not in course_vecs:
        raise HTTPException(status_code=404, detail=f"Option not found: {req.option_id}")

    # Evaluate all
    all_objectives = {}
    for c in courses:
        obj = evaluate_option(id2vec[c.artifact_id], target_centroids, lenses)
        all_objectives[c.artifact_id] = obj

    # Compute Pareto
    keys = list(target_centroids.keys())
    results_list = [{"option_id": oid, "objectives": objs} for oid, objs in all_objectives.items()]
    pareto_idx = pareto_front([r["objectives"] for r in results_list], keys=keys)
    pareto_ids = {results_list[i]["option_id"] for i in pareto_idx}

    # Create explainer
    explainer = Explainer(
        option_embeddings=course_vecs,
        option_texts=course_texts,
        target_centroids=target_centroids,
        target_artifacts=target_artifacts,
        lenses=lenses,
        all_objectives=all_objectives,
        pareto_ids=pareto_ids,
        autonomy_tau=req.config.autonomy_tau,
    )

    # Get explanation
    exp = explainer.explain(req.option_id)

    # Get recourse if not Pareto
    recourse = None
    if req.option_id not in pareto_ids:
        objs = all_objectives[req.option_id]
        if objs:
            weakest = min(objs.items(), key=lambda x: x[1])
            rec = explainer.suggest_recourse(req.option_id, weakest[0])
            recourse = rec.to_dict()

    return ExplainResponse(
        option_id=req.option_id,
        explanation=exp.to_dict(),
        recourse=recourse,
    )
