#!/usr/bin/env python3
"""Constrained vs Unconstrained Pareto analysis on MISIS data.

Compares Pareto fronts with and without constraints:
- Workload constraint (max 30 credits per selection)
- Prerequisite satisfaction
- Minimum market alignment threshold

Quantifies the "price of constraints" — how much the Pareto front shrinks
when real-world limitations are applied.

Usage:
    python -m experiments.scripts.run_constrained_pareto
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from ads_core.eval.pareto import (
    pareto_front,
    constrained_pareto_front,
    compute_hypervolume,
    ParetoResult,
)
from ads_core.eval.constraints import (
    WorkloadConstraint,
    PrerequisiteConstraint,
    ConstraintConfig,
    evaluate_constraints,
)

PARSED_2024 = ROOT / "data" / "raw" / "misis" / "parsed" / "2024_courses.csv"
PREREQ_CSV = ROOT / "data" / "raw" / "misis" / "prerequisites.csv"
REPORT_DIR = ROOT / "experiments" / "reports" / "constrained_pareto"
ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "text.color": "#c9d1d9",
        "grid.color": "#21262d",
        "font.family": "monospace",
        "font.size": 11,
        "font.weight": "bold",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#0d1117",
    })


def load_and_score_courses():
    """Load MISIS courses and score against market/mission/university targets."""
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(PARSED_2024)
    # Build text
    texts = []
    for _, row in df.iterrows():
        parts = []
        name = str(row.get("course_name", "")).strip()
        if name and name != "nan":
            parts.append(name)
        goals = str(row.get("goals", "")).strip()
        if goals and goals != "nan":
            parts.append(goals[:400])
        comps = str(row.get("competency_codes", "")).strip()
        if comps and comps != "nan":
            parts.append(f"Компетенции: {comps}")
        texts.append(". ".join(parts) if parts else "")

    df["text"] = texts
    df = df[df["text"].str.len() > 10].reset_index(drop=True)

    print(f"  {len(df)} courses with text")

    # Encode
    model = SentenceTransformer(ENCODER_NAME)
    embeddings = model.encode(list(df["text"]), show_progress_bar=True, batch_size=128)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Build proxy targets from the data itself (self-referential for demo)
    # "Market" = centroid of courses with high credits (proxy for employable skills)
    credits = df["credits_zet"].fillna(0).astype(float)
    high_credit_mask = credits >= credits.quantile(0.75)
    market_centroid = embeddings[high_credit_mask].mean(axis=0)
    market_centroid /= np.linalg.norm(market_centroid) + 1e-8

    # "Mission" = centroid of all courses (university identity)
    mission_centroid = embeddings.mean(axis=0)
    mission_centroid /= np.linalg.norm(mission_centroid) + 1e-8

    # "University" = centroid of core courses (foundational, low-layer)
    # proxy: courses with most prerequisites pointing to them
    prereqs_df = pd.read_csv(PREREQ_CSV)
    prereqs_24 = prereqs_df[prereqs_df["year"] == 2024]
    prereq_counts = prereqs_24["prerequisite_name"].value_counts()
    core_names = set(prereq_counts.head(50).index)
    core_mask = df["course_name"].isin(core_names)
    if core_mask.sum() > 5:
        univ_centroid = embeddings[core_mask].mean(axis=0)
    else:
        univ_centroid = mission_centroid.copy()
    univ_centroid /= np.linalg.norm(univ_centroid) + 1e-8

    # Score each course against the 3 targets
    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    objectives = []
    for i in range(len(df)):
        objectives.append({
            "market": cosine_sim(embeddings[i], market_centroid),
            "mission": cosine_sim(embeddings[i], mission_centroid),
            "university": cosine_sim(embeddings[i], univ_centroid),
        })

    df["market"] = [o["market"] for o in objectives]
    df["mission"] = [o["mission"] for o in objectives]
    df["university"] = [o["university"] for o in objectives]

    return df, objectives, embeddings


def apply_workload_constraint(df: pd.DataFrame, max_credits: float = 30.0) -> List[bool]:
    """Simple per-course feasibility: flag courses with unreasonable credits."""
    credits = df["credits_zet"].fillna(3).astype(float)
    # A single course > 12 credits is unusual and may indicate data error
    return (credits <= max_credits).tolist()


def apply_min_market_constraint(objectives: List[Dict], threshold: float = 0.3) -> List[bool]:
    """Require minimum market alignment."""
    return [o["market"] >= threshold for o in objectives]


def apply_prerequisite_completeness(df: pd.DataFrame) -> List[bool]:
    """Flag courses whose prerequisites exist in the dataset."""
    prereqs_df = pd.read_csv(PREREQ_CSV)
    prereqs_24 = prereqs_df[prereqs_df["year"] == 2024]
    constraint = PrerequisiteConstraint.from_dataframe(prereqs_24)

    course_names = set(df["course_name"].dropna())
    feasibility = []
    for _, row in df.iterrows():
        name = str(row.get("course_name", ""))
        prereqs = constraint.prerequisite_map.get(name, [])
        # Feasible if all prerequisites exist in our dataset
        unmet = [p for p in prereqs if p not in course_names]
        feasibility.append(len(unmet) == 0)

    return feasibility


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_pareto_comparison(
    objectives: List[Dict],
    unconstrained_idx: List[int],
    constrained_idx: List[int],
    constraint_name: str,
    out_path: Path,
    key_x: str = "market",
    key_y: str = "mission",
):
    """Scatter plot comparing unconstrained and constrained Pareto fronts."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 10))

    xs = [o[key_x] for o in objectives]
    ys = [o[key_y] for o in objectives]

    # All points
    ax.scatter(xs, ys, c="#30363d", s=10, alpha=0.3, label="All courses")

    # Unconstrained Pareto
    unc_x = [xs[i] for i in unconstrained_idx]
    unc_y = [ys[i] for i in unconstrained_idx]
    ax.scatter(unc_x, unc_y, c="#58a6ff", s=40, alpha=0.8,
               label=f"Unconstrained Pareto ({len(unconstrained_idx)})", zorder=3)

    # Constrained Pareto
    con_x = [xs[i] for i in constrained_idx]
    con_y = [ys[i] for i in constrained_idx]
    ax.scatter(con_x, con_y, c="#f0883e", s=60, alpha=0.9, marker="D",
               label=f"Constrained Pareto ({len(constrained_idx)})", zorder=4)

    ax.set_xlabel(key_x.capitalize(), fontsize=12)
    ax.set_ylabel(key_y.capitalize(), fontsize=12)
    ax.set_title(
        f"Constrained vs Unconstrained Pareto Front\n"
        f"Constraint: {constraint_name}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_constraint_cascade(results: Dict[str, Dict], out_path: Path):
    """Bar chart showing Pareto front shrinkage under different constraints."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    names = list(results.keys())
    sizes = [r["pareto_size"] for r in results.values()]
    hvs = [r["hypervolume"] for r in results.values()]
    colors = ["#58a6ff", "#3fb950", "#d2a8ff", "#f0883e", "#f85149"][:len(names)]

    x = range(len(names))
    bars = ax.bar(x, sizes, color=colors, edgecolor="#0d1117", width=0.6)

    for bar, size, hv in zip(bars, sizes, hvs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{size}\nHV={hv:.4f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Pareto front size", fontsize=12)
    ax.set_title("Pareto Front Under Constraint Cascade\n(higher = more options survive)",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_price_of_constraints(results: Dict[str, Dict], out_path: Path):
    """Visualize the 'price of constraints' as HV ratio."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    unconstrained_hv = results["Unconstrained"]["hypervolume"]
    names = [n for n in results if n != "Unconstrained"]
    ratios = [results[n]["hypervolume"] / max(unconstrained_hv, 1e-8) for n in names]
    prices = [1.0 - r for r in ratios]

    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(names)))
    bars = ax.bar(range(len(names)), [p * 100 for p in prices],
                  color=colors, edgecolor="#0d1117", width=0.6)

    for bar, price in zip(bars, prices):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{price:.1%}", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Hypervolume loss (%)", fontsize=12)
    ax.set_title("Price of Constraints\n(% of Pareto quality lost to each constraint)",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Constrained vs Unconstrained Pareto Analysis: MISIS")
    print("=" * 70)
    t0 = time.time()

    fig_dir = REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    keys = ["market", "mission", "university"]

    # ── 1. Load and score ─────────────────────────────────────────────────────
    print("\n[1/4] Loading and scoring courses...")
    df, objectives, embeddings = load_and_score_courses()

    # ── 2. Unconstrained Pareto ───────────────────────────────────────────────
    print("\n[2/4] Computing unconstrained Pareto front...")
    unc_idx = pareto_front(objectives, keys)
    unc_solutions = [objectives[i] for i in unc_idx]
    unc_hv = compute_hypervolume(unc_solutions, keys)
    print(f"  Unconstrained: {len(unc_idx)} Pareto-optimal, HV={unc_hv:.4f}")

    results = {
        "Unconstrained": {
            "pareto_size": len(unc_idx),
            "hypervolume": unc_hv,
            "indices": unc_idx,
        }
    }

    # ── 3. Apply constraints one by one ───────────────────────────────────────
    print("\n[3/4] Computing constrained Pareto fronts...")

    # 3a. Min market threshold
    feas_market = apply_min_market_constraint(objectives, threshold=0.35)
    n_feasible_market = sum(feas_market)
    result_market = constrained_pareto_front(objectives, keys, feas_market)
    hv_market = compute_hypervolume(result_market.solutions, keys) if result_market.solutions else 0
    print(f"  Min-market(0.35): {n_feasible_market} feasible → "
          f"{len(result_market.indices)} Pareto, HV={hv_market:.4f}")
    results["Min Market (≥0.35)"] = {
        "pareto_size": len(result_market.indices),
        "hypervolume": hv_market,
        "feasible_count": n_feasible_market,
        "indices": result_market.indices,
    }

    # 3b. Prerequisite completeness
    feas_prereq = apply_prerequisite_completeness(df)
    n_feasible_prereq = sum(feas_prereq)
    result_prereq = constrained_pareto_front(objectives, keys, feas_prereq)
    hv_prereq = compute_hypervolume(result_prereq.solutions, keys) if result_prereq.solutions else 0
    print(f"  Prerequisites: {n_feasible_prereq} feasible → "
          f"{len(result_prereq.indices)} Pareto, HV={hv_prereq:.4f}")
    results["Prerequisites OK"] = {
        "pareto_size": len(result_prereq.indices),
        "hypervolume": hv_prereq,
        "feasible_count": n_feasible_prereq,
        "indices": result_prereq.indices,
    }

    # 3c. Combined: market + prereqs
    feas_combined = [a and b for a, b in zip(feas_market, feas_prereq)]
    n_feasible_combined = sum(feas_combined)
    result_combined = constrained_pareto_front(objectives, keys, feas_combined)
    hv_combined = compute_hypervolume(result_combined.solutions, keys) if result_combined.solutions else 0
    print(f"  Combined: {n_feasible_combined} feasible → "
          f"{len(result_combined.indices)} Pareto, HV={hv_combined:.4f}")
    results["Market + Prerequisites"] = {
        "pareto_size": len(result_combined.indices),
        "hypervolume": hv_combined,
        "feasible_count": n_feasible_combined,
        "indices": result_combined.indices,
    }

    # 3d. Strict: market(0.4) + prereqs + min mission(0.3)
    feas_strict_market = apply_min_market_constraint(objectives, threshold=0.4)
    feas_mission = [o["mission"] >= 0.3 for o in objectives]
    feas_strict = [a and b and c for a, b, c in zip(feas_strict_market, feas_prereq, feas_mission)]
    n_feasible_strict = sum(feas_strict)
    result_strict = constrained_pareto_front(objectives, keys, feas_strict)
    hv_strict = compute_hypervolume(result_strict.solutions, keys) if result_strict.solutions else 0
    print(f"  Strict: {n_feasible_strict} feasible → "
          f"{len(result_strict.indices)} Pareto, HV={hv_strict:.4f}")
    results["Strict (all)"] = {
        "pareto_size": len(result_strict.indices),
        "hypervolume": hv_strict,
        "feasible_count": n_feasible_strict,
        "indices": result_strict.indices,
    }

    # ── 4. Figures ────────────────────────────────────────────────────────────
    print("\n[4/4] Generating figures...")

    fig_pareto_comparison(
        objectives, unc_idx, result_market.indices,
        "Min Market ≥ 0.35", fig_dir / "F1_market_constraint.png",
    )
    fig_pareto_comparison(
        objectives, unc_idx, result_prereq.indices,
        "Prerequisites Satisfied", fig_dir / "F2_prereq_constraint.png",
    )
    fig_pareto_comparison(
        objectives, unc_idx, result_combined.indices,
        "Market + Prerequisites", fig_dir / "F3_combined_constraint.png",
    )
    fig_pareto_comparison(
        objectives, unc_idx, result_strict.indices,
        "Strict (Market≥0.4 + Prereqs + Mission≥0.3)",
        fig_dir / "F4_strict_constraint.png",
    )

    # Remove indices from results before serializing (not JSON-safe)
    results_json = {}
    for name, r in results.items():
        results_json[name] = {k: v for k, v in r.items() if k != "indices"}

    fig_constraint_cascade(results_json, fig_dir / "F5_cascade.png")
    fig_price_of_constraints(results_json, fig_dir / "F6_price_of_constraints.png")

    # Save summary
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_courses": len(df),
        "objectives": keys,
        "results": results_json,
        "price_of_constraints": {
            name: {
                "hv_loss": round(1.0 - r["hypervolume"] / max(unc_hv, 1e-8), 4),
                "pareto_shrinkage": round(1.0 - r["pareto_size"] / max(len(unc_idx), 1), 4),
            }
            for name, r in results_json.items()
            if name != "Unconstrained"
        },
    }
    json_path = REPORT_DIR / "constrained_pareto_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"  Summary: {json_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Artifacts: {REPORT_DIR}")


if __name__ == "__main__":
    main()
