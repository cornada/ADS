#!/usr/bin/env python
"""Demo script for KT7: Explainability and Recourse.

This script demonstrates:
1. Explaining why an option is on the Pareto front
2. Evidence-based justification with nearest neighbors
3. Autonomy drift breakdown
4. Recourse suggestions for improving weak objectives

Usage:
    python scripts/demo_explain.py
"""
from __future__ import annotations

from pathlib import Path
import sys

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

import numpy as np

from ads_core.ingest.toy_dataset import build_toy_artifacts
from ads_core.data.schemas import ArtifactType
from ads_core.embed.encoder import StubEncoder
from ads_core.lenses.identity import IdentityLens
from ads_core.eval.objectives import centroid, evaluate_option
from ads_core.eval.pareto import pareto_front
from ads_core.eval.constraints import apply_constraints
from ads_core.explain.explainer import Explainer


def main():
    print("=" * 70)
    print("KT7 DEMO: Explainability and Recourse")
    print("=" * 70)
    print()

    # 1) Build toy dataset
    print("[1] Loading toy dataset...")
    artifacts = build_toy_artifacts()

    # 2) Embed with stub encoder
    print("[2] Computing embeddings...")
    enc = StubEncoder(d=64)
    texts = [a.text for a in artifacts]
    vecs = enc.encode(texts)
    id2vec = {artifacts[i].artifact_id: vecs[i] for i in range(len(artifacts))}

    # 3) Build targets
    print("[3] Building target centroids...")
    market_vs = [id2vec[a.artifact_id] for a in artifacts if a.type == ArtifactType.JOB_ROLE]
    mission_vs = [id2vec[a.artifact_id] for a in artifacts if a.type == ArtifactType.MISSION]
    univ_vs = [id2vec[a.artifact_id] for a in artifacts if a.type == ArtifactType.COURSE]
    learner_v = [id2vec[a.artifact_id] for a in artifacts if a.type == ArtifactType.LEARNER_PROFILE][0]

    objectives_list = ["market", "mission", "university", "learner"]
    target_centroids = {
        "market": centroid(market_vs),
        "mission": centroid(mission_vs),
        "university": centroid(univ_vs),
        "learner": learner_v / (np.linalg.norm(learner_v) + 1e-8),
    }

    lenses = {k: IdentityLens(lens_id=f"identity:{k}") for k in target_centroids}

    # Build target artifacts for evidence
    target_artifacts = {
        "market": [(a.artifact_id, a.text, id2vec[a.artifact_id]) for a in artifacts if a.type == ArtifactType.JOB_ROLE],
        "mission": [(a.artifact_id, a.text, id2vec[a.artifact_id]) for a in artifacts if a.type == ArtifactType.MISSION],
        "university": [(a.artifact_id, a.text, id2vec[a.artifact_id]) for a in artifacts if a.type == ArtifactType.COURSE],
        "learner": [(a.artifact_id, a.text, id2vec[a.artifact_id]) for a in artifacts if a.type == ArtifactType.LEARNER_PROFILE],
    }

    # 4) Evaluate courses
    print("[4] Evaluating course options...")
    courses = [a for a in artifacts if a.type == ArtifactType.COURSE]
    course_vecs = {a.artifact_id: id2vec[a.artifact_id] for a in courses}
    course_texts = {a.artifact_id: a.text for a in courses}

    autonomy_tau = 0.25
    results = []
    for c in courses:
        obj = evaluate_option(id2vec[c.artifact_id], target_centroids, lenses)
        cons = apply_constraints(obj, tau=autonomy_tau)
        results.append({
            "option_id": c.artifact_id,
            "objectives": obj,
            "feasible": cons["feasible"],
        })

    # 5) Compute Pareto front
    print("[5] Computing Pareto front...")
    pareto_idx = pareto_front([r["objectives"] for r in results], keys=objectives_list)
    pareto_ids = {results[i]["option_id"] for i in pareto_idx}

    all_objectives = {r["option_id"]: r["objectives"] for r in results}

    print()
    print("-" * 70)
    print("PARETO FRONT:")
    print("-" * 70)
    for idx in pareto_idx:
        r = results[idx]
        print(f"  - {r['option_id']}: feasible={r['feasible']}")
        for k, v in r["objectives"].items():
            print(f"      {k}: {v:.3f}")
    print()

    # 6) Create explainer
    print("[6] Initializing explainer...")
    explainer = Explainer(
        option_embeddings=course_vecs,
        option_texts=course_texts,
        target_centroids=target_centroids,
        target_artifacts=target_artifacts,
        lenses=lenses,
        all_objectives=all_objectives,
        pareto_ids=pareto_ids,
        autonomy_tau=autonomy_tau,
    )

    # 7) Demo: Explain a Pareto-optimal option
    print()
    print("=" * 70)
    print("EXPLANATION: Pareto-optimal option")
    print("=" * 70)

    # Pick first Pareto option
    pareto_option = list(pareto_ids)[0]
    exp = explainer.explain(pareto_option)

    print(f"\nOption: {exp.option_id}")
    print(f"Text: {exp.option_text}")
    print(f"\nStatus: {exp.pareto_status.upper()}")
    print(f"Reason: {exp.pareto_reason}")
    print(f"\nSummary: {exp.overall_summary}")

    print("\nObjective Breakdown:")
    for obj_name, obj_exp in exp.objectives.items():
        print(f"\n  {obj_name.upper()}:")
        print(f"    Score: {obj_exp.score:.3f} (rank #{obj_exp.rank}, {obj_exp.percentile:.0f}th percentile)")
        print(f"    {obj_exp.interpretation}")
        if obj_exp.evidence:
            print("    Evidence:")
            for ev in obj_exp.evidence[:2]:
                print(f"      - {ev.artifact_id}: {ev.contribution} (sim={ev.similarity:.3f})")
                print(f"        \"{ev.text[:60]}...\"")

    if exp.drift:
        print("\nAutonomy Drift Analysis:")
        print(f"  Total drift: {exp.drift.total_drift:.3f}")
        print(f"  Market drift: {exp.drift.market_drift:.3f}")
        print(f"  University drift: {exp.drift.university_drift:.3f}")
        print(f"  Learner score: {exp.drift.learner_score:.3f}")
        print(f"  Within threshold ({exp.drift.threshold}): {exp.drift.within_threshold}")
        print(f"  Interpretation: {exp.drift.interpretation}")

    # 8) Demo: Recourse for a dominated/infeasible option
    print()
    print("=" * 70)
    print("RECOURSE: Improving a weak objective")
    print("=" * 70)

    # Find an option not on Pareto or with lowest learner score
    sorted_by_learner = sorted(results, key=lambda r: r["objectives"].get("learner", 0))
    weak_option = sorted_by_learner[0]["option_id"]
    weak_score = sorted_by_learner[0]["objectives"].get("learner", 0)

    print(f"\nOption with weak learner alignment: {weak_option}")
    print(f"Current learner score: {weak_score:.3f}")

    rec = explainer.suggest_recourse(weak_option, "learner", top_k=3, min_improvement=0.01)

    print(f"\n{rec.summary}")

    if rec.best_alternative:
        best = rec.best_alternative
        print(f"\nBest Alternative: {best.option_id}")
        print(f"  Text: {best.option_text[:80]}...")
        print(f"  Improvement: +{best.improvement:.3f} (new score: {best.new_score:.3f})")
        print(f"  Net benefit: {best.net_benefit:.3f}")
        print("  Trade-offs:")
        for obj, delta in best.trade_offs.items():
            direction = "+" if delta >= 0 else ""
            print(f"    {obj}: {direction}{delta:.3f}")
        print(f"\n  Recommendation: {best.recommendation}")

    print("\nAll alternatives:")
    for i, alt in enumerate(rec.alternatives, 1):
        print(f"  {i}. {alt.option_id}: +{alt.improvement:.3f} improvement, net={alt.net_benefit:.3f}")

    # 9) Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
KT7 Explainability Features:
  1. Evidence-based explanations: Show which artifacts drive objective scores
  2. Ranking context: Position relative to other options (rank, percentile)
  3. Autonomy drift analysis: Breakdown of learner vs external stakeholder alignment
  4. Pareto status explanation: Why an option is/isn't on the Pareto front
  5. Recourse suggestions: Alternatives that improve weak objectives with trade-off analysis

This supports:
  - Transparent recommendations (users understand why options are ranked)
  - Contestable decisions (users can challenge based on evidence)
  - Actionable guidance (concrete alternatives for improvement)
""")


if __name__ == "__main__":
    main()
