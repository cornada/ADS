"""Generic dataset pipeline for ADS experiments.

Works with any dataset from the registry (toy, mit, ucb, asu).
Provides the same evaluator interface across all datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone
import json
import numpy as np

from ads_core.data.storage import LocalArtifactStore
from ads_core.data.schemas import Artifact, ArtifactType
from ads_core.datasets.registry import load_dataset, DatasetBundle
from ads_core.embed.encoder import StubEncoder
from ads_core.embed.sentence_transformers_encoder import SentenceTransformersEncoder
from ads_core.embed.cache import DiskEmbeddingCache
from ads_core.lenses.identity import IdentityLens
from ads_core.lenses.diagonal import DiagonalLens
from ads_core.lenses.learn_lenses import learn_one_vs_rest_diagonal_weights
from ads_core.eval.objectives import centroid, evaluate_option
from ads_core.eval.pareto import compute_pareto_with_feasibility
from ads_core.eval.constraints import apply_constraints


@dataclass
class DatasetRunOutputs:
    """Outputs from a dataset experiment run."""
    run_dir: Path
    dataset_id: str
    results_json: Path
    pareto_json: Path
    artifact_count: int
    pareto_count: int


def _select_encoder(embedding_cfg: dict):
    """Select encoder based on config."""
    kind = embedding_cfg.get("kind", "stub")
    if kind == "stub":
        return StubEncoder(d=int(embedding_cfg.get("d", 64)))
    if kind == "sbert":
        return SentenceTransformersEncoder(
            model_name=str(embedding_cfg["model_name"]),
            device=embedding_cfg.get("device"),
        )
    raise ValueError(f"Unknown embedding kind: {kind}")


def _build_lenses(lenses_cfg: dict, d: int, id2vec: dict, artifacts: List[Artifact]):
    """Build lens transformations based on config."""
    kind = lenses_cfg.get("kind", "identity")

    if kind == "identity":
        return {"default": IdentityLens(lens_id="identity:default")}

    if kind == "diagonal":
        return {"default": DiagonalLens(lens_id="diagonal:default", weights=np.ones(d))}

    if kind == "learned":
        # Learn per-stakeholder lenses
        course_arts = [a for a in artifacts if a.type == ArtifactType.COURSE]
        job_arts = [a for a in artifacts if a.type == ArtifactType.JOB_ROLE]

        if course_arts and job_arts:
            course_vecs = np.array([id2vec[a.artifact_id] for a in course_arts])
            job_vecs = np.array([id2vec[a.artifact_id] for a in job_arts])

            weights, _ = learn_one_vs_rest_diagonal_weights(
                X=np.vstack([course_vecs, job_vecs]),
                y=[0] * len(course_vecs) + [1] * len(job_vecs),
            )
            return {"default": DiagonalLens(lens_id="learned:default", weights=weights)}

        # Fallback to identity
        return {"default": IdentityLens(lens_id="identity:default")}

    raise ValueError(f"Unknown lenses kind: {kind}")


def _get_target_artifacts(
    artifacts: List[Artifact],
    target_type: str,
) -> List[Artifact]:
    """Get artifacts for a target type."""
    type_map = {
        "market": [ArtifactType.JOB_ROLE, ArtifactType.SKILL],
        "mission": [ArtifactType.MISSION],
        "university": [ArtifactType.COURSE],
        "outcomes": [ArtifactType.OUTCOME_MAJOR, ArtifactType.OUTCOME_SUMMARY],
    }

    target_types = type_map.get(target_type, [])
    return [a for a in artifacts if a.type in target_types]


def _create_learner_profile(dataset_id: str) -> str:
    """Create a default learner profile text for a dataset."""
    profiles = {
        "toy": "Computer science student interested in AI and software engineering.",
        "mit": "MIT student interested in algorithms, machine learning, and systems.",
        "ucb": "UC Berkeley student interested in computer science and data science.",
        "asu": "ASU student interested in engineering and technology careers.",
    }
    return profiles.get(dataset_id, "Student interested in technology and career success.")


def run_dataset(
    dataset_id: str,
    out_dir: Path,
    seed: int,
    embedding_cfg: dict,
    lenses_cfg: dict,
    objectives: List[str],
    autonomy_tau: float,
    data_dir: Optional[Path] = None,
) -> DatasetRunOutputs:
    """Run evaluation pipeline on a dataset.

    Args:
        dataset_id: Dataset to load (toy, mit, ucb, asu)
        out_dir: Output directory for results
        seed: Random seed
        embedding_cfg: Embedding configuration
        lenses_cfg: Lens configuration
        objectives: List of objectives to optimize
        autonomy_tau: Autonomy drift threshold
        data_dir: Base data directory (defaults to project root)

    Returns:
        DatasetRunOutputs with paths to results
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # 1) Load dataset
    bundle = load_dataset(dataset_id, data_dir)
    artifacts = bundle.artifacts

    # Store artifacts
    store = LocalArtifactStore(root=out_dir / "store")
    if store.root.exists():
        for p in store.root.glob("*"):
            p.unlink()
    store.save_artifacts(artifacts)
    store.write_meta({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_id,
        "seed": seed,
        "artifact_count": len(artifacts),
    })

    # 2) Compute embeddings
    enc = _select_encoder(embedding_cfg)
    texts = [a.text for a in artifacts]
    cache = DiskEmbeddingCache(root=out_dir / "cache_embeddings", model_id=enc.model_id)
    vecs = cache.get_or_compute(texts, enc.encode, verbose=True)

    id2vec = {artifacts[i].artifact_id: vecs[i] for i in range(len(artifacts))}
    d = vecs.shape[1]

    # 3) Build lenses
    lens_result = _build_lenses(lenses_cfg, d, id2vec, artifacts)
    default_lens = lens_result["default"]

    # 4) Build target centroids
    target_centroids: Dict[str, np.ndarray] = {}

    # Market target (jobs/skills if available, else use outcomes)
    if "market" in objectives:
        market_arts = _get_target_artifacts(artifacts, "market")
        if not market_arts:
            # Fall back to outcomes as market proxy
            market_arts = _get_target_artifacts(artifacts, "outcomes")
        if market_arts:
            market_vs = [id2vec[a.artifact_id] for a in market_arts]
            target_centroids["market"] = centroid(market_vs)

    # Mission target
    if "mission" in objectives:
        mission_arts = _get_target_artifacts(artifacts, "mission")
        if mission_arts:
            mission_vs = [id2vec[a.artifact_id] for a in mission_arts]
            target_centroids["mission"] = centroid(mission_vs)

    # University target (courses)
    if "university" in objectives:
        univ_arts = _get_target_artifacts(artifacts, "university")
        if univ_arts:
            univ_vs = [id2vec[a.artifact_id] for a in univ_arts]
            target_centroids["university"] = centroid(univ_vs)

    # Learner target
    if "learner" in objectives:
        # Create synthetic learner profile
        learner_text = _create_learner_profile(dataset_id)
        learner_v = enc.encode([learner_text])[0]
        target_centroids["learner"] = learner_v / (np.linalg.norm(learner_v) + 1e-8)

    # 5) Get options (courses for evaluation)
    option_arts = [a for a in artifacts if a.type == ArtifactType.COURSE]
    if not option_arts:
        # Fall back to all artifacts as options
        option_arts = artifacts

    # 6) Build per-objective lens dict (same lens for all objectives)
    lenses_dict = {k: default_lens for k in target_centroids.keys()}

    # 7) Evaluate each option with constraints
    results = []
    for art in option_arts:
        v = id2vec[art.artifact_id]
        obj_scores = evaluate_option(v, target_centroids, lenses_dict)

        # Apply autonomy constraint if learner objective is present
        if "learner" in obj_scores:
            cons = apply_constraints(obj_scores, tau=autonomy_tau)
            feasible = cons["feasible"]
            violations = cons["violations"]
        else:
            feasible = True
            violations = {}

        results.append({
            "option_id": art.artifact_id,
            "objectives": obj_scores,
            "feasible": feasible,
            "violations": violations,
        })

    # 8) Compute Pareto front
    objective_keys = list(target_centroids.keys())
    pareto_result = compute_pareto_with_feasibility(results, objective_keys)

    # Extract pareto-optimal results in the format expected by report builder
    pareto_list = [results[i] for i in pareto_result.indices]

    # 9) Save outputs
    results_json = out_dir / "results.json"
    pareto_json = out_dir / "pareto.json"

    results_json.write_text(json.dumps({
        "dataset": dataset_id,
        "encoder": enc.model_id,
        "objectives": objective_keys,
        "autonomy_tau": autonomy_tau,
        "results": results,
    }, indent=2), encoding="utf-8")

    pareto_json.write_text(json.dumps({
        "keys": objective_keys,
        "pareto": pareto_list,
    }, indent=2), encoding="utf-8")

    return DatasetRunOutputs(
        run_dir=out_dir,
        dataset_id=dataset_id,
        results_json=results_json,
        pareto_json=pareto_json,
        artifact_count=len(artifacts),
        pareto_count=len(pareto_result.indices),
    )
