"""Encoder ablation: compare MiniLM-L12 (384d) vs E5-large (1024d) vs BGE-large (1024d).

Measures impact of model choice on:
1. Pareto front composition (which courses are Pareto-optimal)
2. Objective score distributions
3. Pairwise Jaccard similarity of Pareto sets
4. Embedding space geometry (inter/intra cluster distances)

Usage:
    python -m experiments.scripts.encoder_ablation
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ENCODERS = {
    "MiniLM-L12 (384d)": {
        "kind": "sbert",
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
    "E5-large (1024d)": {
        "kind": "sbert",
        "model_name": "intfloat/multilingual-e5-large-instruct",
    },
    "BGE-large (1024d)": {
        "kind": "sbert",
        "model_name": "BAAI/bge-large-en-v1.5",
    },
}

REPORT_DIR = Path("experiments/reports/encoder_ablation")


def load_texts() -> Tuple[List[str], List[str]]:
    """Load course and job texts from toy dataset."""
    from ads_core.ingest.toy_dataset import build_toy_artifacts
    from ads_core.data.schemas import ArtifactType

    artifacts = build_toy_artifacts()
    courses = [a for a in artifacts if a.type == ArtifactType.COURSE]
    jobs = [a for a in artifacts if a.type == ArtifactType.JOB_ROLE]
    return [c.text for c in courses], [j.text for j in jobs], courses, jobs


def encode_with_cache(encoder_name: str, cfg: dict, texts: List[str]) -> np.ndarray:
    """Encode texts using shared global cache."""
    from ads_core.embed.encoder_factory import select_encoder
    from ads_core.embed.cache import DiskEmbeddingCache

    enc = select_encoder(cfg)
    cache_dir = Path.home() / ".cache" / "ads" / "embeddings"
    cache = DiskEmbeddingCache(root=cache_dir, model_id=enc.model_id)
    t0 = time.time()
    vecs = cache.get_or_compute(texts, enc.encode, verbose=True)
    dt = time.time() - t0
    logger.info("%s: encoded %d texts in %.1fs (dim=%d)", encoder_name, len(texts), dt, vecs.shape[1])
    return vecs


def compute_pareto_front(scores: np.ndarray) -> List[int]:
    """Simple Pareto front (maximize all objectives)."""
    n = scores.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                is_pareto[i] = False
                break
    return [i for i in range(n) if is_pareto[i]]


def compute_objectives(course_vecs: np.ndarray, job_vecs: np.ndarray) -> np.ndarray:
    """Compute 2 objectives: market alignment + diversity."""
    # Objective 1: cosine similarity to job centroid
    job_centroid = job_vecs.mean(axis=0)
    job_centroid /= np.linalg.norm(job_centroid) + 1e-8
    course_norms = course_vecs / (np.linalg.norm(course_vecs, axis=1, keepdims=True) + 1e-8)
    market_scores = course_norms @ job_centroid

    # Objective 2: distance from course centroid (diversity)
    course_centroid = course_vecs.mean(axis=0)
    course_centroid /= np.linalg.norm(course_centroid) + 1e-8
    diversity_scores = 1.0 - (course_norms @ course_centroid)

    return np.column_stack([market_scores, diversity_scores])


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def run_ablation():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    course_texts, job_texts, courses, jobs = load_texts()
    all_texts = course_texts + job_texts
    n_courses = len(course_texts)

    results: Dict[str, Any] = {}
    pareto_sets: Dict[str, set] = {}

    for name, cfg in ENCODERS.items():
        logger.info("=== %s ===", name)
        try:
            vecs = encode_with_cache(name, cfg, all_texts)
            course_vecs = vecs[:n_courses]
            job_vecs = vecs[n_courses:]

            objectives = compute_objectives(course_vecs, job_vecs)
            pareto_idx = compute_pareto_front(objectives)
            pareto_ids = {courses[i].artifact_id for i in pareto_idx}
            pareto_sets[name] = pareto_ids

            # Embedding space stats
            dists = np.linalg.norm(course_vecs[:, None] - course_vecs[None, :], axis=2)
            np.fill_diagonal(dists, np.nan)

            results[name] = {
                "dim": int(vecs.shape[1]),
                "n_courses": n_courses,
                "n_pareto": len(pareto_idx),
                "pareto_ids": sorted(pareto_ids),
                "market_scores": {"mean": float(objectives[:, 0].mean()), "std": float(objectives[:, 0].std())},
                "diversity_scores": {"mean": float(objectives[:, 1].mean()), "std": float(objectives[:, 1].std())},
                "pairwise_dist": {"mean": float(np.nanmean(dists)), "std": float(np.nanstd(dists))},
            }
            logger.info("  Pareto front: %d/%d courses", len(pareto_idx), n_courses)

        except Exception as e:
            logger.error("Failed for %s: %s", name, e)
            results[name] = {"error": str(e)}

    # Pairwise Jaccard similarity
    names = list(pareto_sets.keys())
    jaccard_matrix = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i <= j:
                j_val = jaccard(pareto_sets[a], pareto_sets[b])
                jaccard_matrix[f"{a} vs {b}"] = round(j_val, 3)

    results["jaccard_similarity"] = jaccard_matrix

    # Save results
    out_path = REPORT_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Print summary
    print("\n" + "=" * 60)
    print("ENCODER ABLATION RESULTS")
    print("=" * 60)
    for name in ENCODERS:
        r = results.get(name, {})
        if "error" in r:
            print(f"\n{name}: ERROR - {r['error']}")
            continue
        print(f"\n{name} (dim={r['dim']}):")
        print(f"  Pareto: {r['n_pareto']}/{r['n_courses']} courses")
        print(f"  Market: {r['market_scores']['mean']:.3f} +/- {r['market_scores']['std']:.3f}")
        print(f"  Diversity: {r['diversity_scores']['mean']:.3f} +/- {r['diversity_scores']['std']:.3f}")
        print(f"  Pairwise dist: {r['pairwise_dist']['mean']:.3f} +/- {r['pairwise_dist']['std']:.3f}")

    print(f"\nJaccard similarity between Pareto sets:")
    for pair, val in jaccard_matrix.items():
        print(f"  {pair}: {val}")


if __name__ == "__main__":
    run_ablation()
