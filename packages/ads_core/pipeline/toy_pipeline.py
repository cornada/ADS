from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timezone
import json
import numpy as np

from ads_core.data.storage import LocalArtifactStore
from ads_core.data.schemas import ArtifactType
from ads_core.ingest.toy_dataset import build_toy_artifacts
from ads_core.embed.encoder import StubEncoder
from ads_core.embed.sentence_transformers_encoder import SentenceTransformersEncoder
from ads_core.embed.cache import DiskEmbeddingCache
from ads_core.lenses.identity import IdentityLens
from ads_core.lenses.diagonal import DiagonalLens
from ads_core.lenses.learn_lenses import learn_one_vs_rest_diagonal_weights
from ads_core.eval.objectives import centroid, evaluate_option
from ads_core.eval.pareto import pareto_front, compute_pareto_with_feasibility
from ads_core.eval.constraints import apply_constraints
from ads_core.plan.candidates import topk_by_similarity
from ads_core.plan.pathway import plan_pathways, pathways_to_json
from ads_core.eval.stability import run_stability_analysis
from ads_core.eval.constraints import autonomy_drift_detailed
from ads_core.explain.explainer import Explainer


@dataclass
class RunOutputs:
    run_dir: Path
    results_json: Path
    pareto_json: Path
    pathways_json: Path = None
    stability_json: Path = None
    ethics_json: Path = None
    explain_json: Path = None


def _select_encoder(embedding_cfg: dict):
    kind = embedding_cfg.get("kind", "stub")
    if kind == "stub":
        return StubEncoder(d=int(embedding_cfg.get("d", 64)))
    if kind == "sbert":
        return SentenceTransformersEncoder(
            model_name=str(embedding_cfg["model_name"]),
            device=embedding_cfg.get("device"),
        )
    raise ValueError(f"Unknown embedding kind: {kind}")


def run_toy(
    out_dir: Path,
    seed: int,
    embedding_cfg: dict,
    lenses_cfg: dict,
    objectives: List[str],
    autonomy_tau: float,
    topk: int = 5,
    stability_runs: int = 0,
    stability_noise: float = 0.01,
) -> RunOutputs:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Ingest toy dataset
    store = LocalArtifactStore(root=out_dir / "store")
    artifacts = build_toy_artifacts()
    # Overwrite store deterministically for toy runs
    if store.root.exists():
        # wipe
        for p in store.root.glob("*"):
            p.unlink()
    store.save_artifacts(artifacts)
    store.write_meta({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "toy",
        "seed": seed,
    })

    # 2) Embeddings with cache
    enc = _select_encoder(embedding_cfg)
    texts = [a.text for a in artifacts]
    cache = DiskEmbeddingCache(root=out_dir / "cache_embeddings", model_id=enc.model_id)
    vecs = cache.get_or_compute(texts, enc.encode, verbose=True)  # [n,d]

    id2vec = {artifacts[i].artifact_id: vecs[i] for i in range(len(artifacts))}

    # 3) Build target sets (toy mapping)
    # market targets = JOB_ROLE artifacts
    market_vs = [id2vec[a.artifact_id] for a in artifacts if a.type == ArtifactType.JOB_ROLE]
    mission_vs = [id2vec[a.artifact_id] for a in artifacts if a.type == ArtifactType.MISSION]
    univ_vs = [id2vec[a.artifact_id] for a in artifacts if a.type == ArtifactType.COURSE]  # proxy
    learner_v = [id2vec[a.artifact_id] for a in artifacts if a.type == ArtifactType.LEARNER_PROFILE][0]

    target_centroids: Dict[str, np.ndarray] = {}
    if "market" in objectives: target_centroids["market"] = centroid(market_vs)
    if "mission" in objectives: target_centroids["mission"] = centroid(mission_vs)
    if "university" in objectives: target_centroids["university"] = centroid(univ_vs)
    if "learner" in objectives: target_centroids["learner"] = learner_v / (np.linalg.norm(learner_v) + 1e-8)

    # 4) Lenses
    lens_mode = lenses_cfg.get("mode", "identity")

    lenses = {}
    if lens_mode == "identity":
        lenses = {k: IdentityLens(lens_id=f"identity:{k}") for k in target_centroids.keys()}
    elif lens_mode == "diagonal":
        # MVP: use uniform weights or random seed weights; here uniform==ones
        d = next(iter(id2vec.values())).shape[0]
        w = np.ones((d,), dtype=np.float32)
        lenses = {k: DiagonalLens(lens_id=f"diag:{k}", weights=w) for k in target_centroids.keys()}
    elif lens_mode == "learned":
        # Learn diagonal weights from toy stakeholder corpora (labels by type)
        # Labels: market for job roles, mission for mission, university for courses, learner for learner profile
        X = []
        y = []
        for a in artifacts:
            if a.type == ArtifactType.JOB_ROLE:
                X.append(id2vec[a.artifact_id]); y.append("market")
            elif a.type == ArtifactType.MISSION:
                X.append(id2vec[a.artifact_id]); y.append("mission")
            elif a.type == ArtifactType.COURSE:
                X.append(id2vec[a.artifact_id]); y.append("university")
            elif a.type == ArtifactType.LEARNER_PROFILE:
                X.append(id2vec[a.artifact_id]); y.append("learner")
        X = np.stack(X, axis=0)
        learned = learn_one_vs_rest_diagonal_weights(X, y, stakeholders=list(target_centroids.keys()), seed=seed)
        lenses = {k: DiagonalLens(lens_id=f"learned:{k}", weights=learned.weights[k]) for k in target_centroids.keys()}
    else:
        raise ValueError(f"Unknown lens mode: {lens_mode}")

    # 5) Candidate options: courses topK by similarity to learner
    course_vecs = {a.artifact_id: id2vec[a.artifact_id] for a in artifacts if a.type == ArtifactType.COURSE}
    top = topk_by_similarity(learner_v, course_vecs, k=topk)
    candidate_ids = [cid for cid, _ in top]

    # 6) Evaluate each candidate option with ethics metrics
    results = []
    for cid in candidate_ids:
        obj = evaluate_option(id2vec[cid], target_centroids, lenses)
        cons = apply_constraints(obj, tau=autonomy_tau)
        drift_details = autonomy_drift_detailed(obj)
        results.append({
            "option_id": cid,
            "objectives": obj,
            "feasible": cons["feasible"],
            "violations": cons["violations"],
            "autonomy_drift": drift_details,
        })

    # 7) Pareto front over enabled objectives
    keys = list(target_centroids.keys())
    pareto_idx = pareto_front([r["objectives"] for r in results], keys=keys)
    pareto = [results[i] for i in pareto_idx]

    # 8) Pathway planning (depth 2-3)
    def constraint_fn(obj):
        return apply_constraints(obj, tau=autonomy_tau)

    pathways = plan_pathways(
        course_embeddings=course_vecs,
        target_centroids=target_centroids,
        lenses=lenses,
        max_depth=2,
        beam_width=10,
        aggregation="mean",
        constraint_fn=constraint_fn,
    )

    # 9) Save outputs
    results_path = out_dir / "results.json"
    pareto_path = out_dir / "pareto.json"
    pathways_path = out_dir / "pathways.json"

    results_path.write_text(json.dumps({
        "encoder": enc.model_id,
        "results": results,
        "constraint_tau": autonomy_tau,
    }, indent=2), encoding="utf-8")

    pareto_path.write_text(json.dumps({
        "keys": keys,
        "pareto": pareto,
        "constraint_tau": autonomy_tau,
    }, indent=2), encoding="utf-8")

    pathways_path.write_text(json.dumps({
        "keys": keys,
        "pathways": pathways_to_json(pathways),
        "max_depth": 2,
        "constraint_tau": autonomy_tau,
    }, indent=2), encoding="utf-8")

    # 10) Ethics/autonomy summary
    ethics_path = out_dir / "ethics.json"
    ethics_summary = {
        "constraint_tau": autonomy_tau,
        "options_evaluated": len(results),
        "feasible_count": sum(1 for r in results if r["feasible"]),
        "infeasible_count": sum(1 for r in results if not r["feasible"]),
        "drift_summary": {
            "mean": float(np.mean([r["autonomy_drift"]["total"] for r in results])),
            "max": float(np.max([r["autonomy_drift"]["total"] for r in results])),
            "min": float(np.min([r["autonomy_drift"]["total"] for r in results])),
        },
        "per_option": {
            r["option_id"]: {
                "drift_total": r["autonomy_drift"]["total"],
                "market_drift": r["autonomy_drift"]["market_drift"],
                "university_drift": r["autonomy_drift"]["university_drift"],
                "feasible": r["feasible"],
            }
            for r in results
        },
    }
    ethics_path.write_text(json.dumps(ethics_summary, indent=2), encoding="utf-8")

    # 11) Stability evaluation (if enabled)
    stability_path = None
    if stability_runs > 0:
        stability_path = out_dir / "stability.json"
        stability_result = run_stability_analysis(
            embeddings=course_vecs,
            target_centroids=target_centroids,
            lenses=lenses,
            objective_keys=keys,
            constraint_fn=constraint_fn,
            n_runs=stability_runs,
            noise_scale=stability_noise,
            seed=seed,
        )
        stability_path.write_text(json.dumps(stability_result.to_dict(), indent=2), encoding="utf-8")
        print(f"[stability] Jaccard={stability_result.jaccard_mean:.3f} +/- {stability_result.jaccard_std:.3f}")

    # 12) Explainability and recourse
    # Build target artifacts for evidence
    target_artifacts = {}
    for a in artifacts:
        if a.type == ArtifactType.JOB_ROLE and "market" in target_centroids:
            if "market" not in target_artifacts:
                target_artifacts["market"] = []
            target_artifacts["market"].append((a.artifact_id, a.text, id2vec[a.artifact_id]))
        elif a.type == ArtifactType.MISSION and "mission" in target_centroids:
            if "mission" not in target_artifacts:
                target_artifacts["mission"] = []
            target_artifacts["mission"].append((a.artifact_id, a.text, id2vec[a.artifact_id]))
        elif a.type == ArtifactType.COURSE and "university" in target_centroids:
            if "university" not in target_artifacts:
                target_artifacts["university"] = []
            target_artifacts["university"].append((a.artifact_id, a.text, id2vec[a.artifact_id]))
        elif a.type == ArtifactType.LEARNER_PROFILE and "learner" in target_centroids:
            if "learner" not in target_artifacts:
                target_artifacts["learner"] = []
            target_artifacts["learner"].append((a.artifact_id, a.text, id2vec[a.artifact_id]))

    # Get option texts
    option_texts = {a.artifact_id: a.text for a in artifacts if a.type == ArtifactType.COURSE}

    # Build all objectives dict
    all_objectives = {r["option_id"]: r["objectives"] for r in results}

    # Get Pareto IDs
    pareto_ids = {r["option_id"] for r in pareto}

    # Create explainer
    explainer = Explainer(
        option_embeddings=course_vecs,
        option_texts=option_texts,
        target_centroids=target_centroids,
        target_artifacts=target_artifacts,
        lenses=lenses,
        all_objectives=all_objectives,
        pareto_ids=pareto_ids,
        autonomy_tau=autonomy_tau,
    )

    # Generate explanations for all options
    explanations = []
    for r in results:
        try:
            exp = explainer.explain(r["option_id"])
            explanations.append(exp.to_dict())
        except Exception as e:
            explanations.append({"option_id": r["option_id"], "error": str(e)})

    # Generate recourse for non-Pareto options with weak objectives
    recourse_suggestions = []
    for r in results:
        if r["option_id"] not in pareto_ids:
            # Find weakest objective
            objs = r["objectives"]
            if objs:
                weakest = min(objs.items(), key=lambda x: x[1])
                try:
                    rec = explainer.suggest_recourse(r["option_id"], weakest[0])
                    recourse_suggestions.append(rec.to_dict())
                except Exception as e:
                    recourse_suggestions.append({"option_id": r["option_id"], "error": str(e)})

    explain_path = out_dir / "explain.json"
    explain_path.write_text(json.dumps({
        "explanations": explanations,
        "recourse": recourse_suggestions,
        "autonomy_tau": autonomy_tau,
    }, indent=2), encoding="utf-8")

    return RunOutputs(
        run_dir=out_dir,
        results_json=results_path,
        pareto_json=pareto_path,
        pathways_json=pathways_path,
        stability_json=stability_path,
        ethics_json=ethics_path,
        explain_json=explain_path,
    )
