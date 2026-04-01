#!/usr/bin/env python3
"""Topological Data Analysis on MISIS curriculum embeddings.

Computes persistent homology on course embedding point clouds,
comparing topology across FGOS directions and against market spaces.

Usage:
    python -m experiments.scripts.run_tda_analysis

Output:
    experiments/reports/tda_analysis/
        tda_summary.json
        persistence_diagram_*.png
        betti_comparison.png
        TDA_REPORT.html
"""
from __future__ import annotations

import json
import os
import sys
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))

from ads_core.topology.persistent_homology import (
    compute_persistence,
    compare_topologies,
    reduce_embeddings,
    TopologicalSummary,
)


OUT_DIR = Path("experiments/reports/tda_analysis")

plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#1e1e2e",
    "axes.edgecolor": "#cdd6f4",
    "axes.labelcolor": "#cdd6f4",
    "text.color": "#cdd6f4",
    "xtick.color": "#cdd6f4",
    "ytick.color": "#cdd6f4",
    "font.size": 11,
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})


def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def encode_texts(texts: List[str]) -> np.ndarray:
    """Encode texts using multilingual sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return model.encode(texts, show_progress_bar=True, batch_size=64)


def plot_persistence_diagram(
    summary: TopologicalSummary,
    title: str,
    out_path: Path,
) -> str:
    """Plot persistence diagram (birth vs death)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {0: "#89b4fa", 1: "#a6e3a1", 2: "#fab387"}
    dim_labels = {0: "H₀ (components)", 1: "H₁ (loops)", 2: "H₂ (voids)"}

    max_val = 0
    for dim in range(summary.max_homology_dim + 1):
        feats = summary.features_by_dim(dim)
        if not feats:
            continue
        births = [f.birth for f in feats]
        deaths = [f.death for f in feats]
        max_val = max(max_val, max(deaths) if deaths else 0)
        ax.scatter(
            births, deaths,
            c=colors.get(dim, "#cdd6f4"),
            alpha=0.6,
            s=30,
            label=f"{dim_labels.get(dim, f'H{dim}')} ({len(feats)})",
            edgecolors="white",
            linewidth=0.3,
        )

    # Diagonal line (birth = death)
    if max_val > 0:
        ax.plot([0, max_val * 1.1], [0, max_val * 1.1],
                "k--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_persistence_barcode(
    summary: TopologicalSummary,
    title: str,
    out_path: Path,
    max_bars: int = 50,
) -> str:
    """Plot persistence barcode (horizontal bars showing lifetime)."""
    colors = {0: "#89b4fa", 1: "#a6e3a1", 2: "#fab387"}

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = 0

    for dim in range(summary.max_homology_dim + 1):
        feats = sorted(summary.features_by_dim(dim),
                       key=lambda f: f.persistence, reverse=True)[:max_bars]
        color = colors.get(dim, "#cdd6f4")
        for f in feats:
            ax.barh(y_pos, f.persistence, left=f.birth,
                    height=0.8, color=color, alpha=0.7)
            y_pos += 1

    ax.set_xlabel("Scale (distance)")
    ax.set_ylabel("Feature index")
    ax.set_title(title)
    ax.grid(True, alpha=0.15, axis="x")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[d], label=f"H{d}")
        for d in range(summary.max_homology_dim + 1)
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_betti_comparison(
    summaries: Dict[str, TopologicalSummary],
    out_path: Path,
) -> str:
    """Compare Betti numbers across subsets."""
    labels = list(summaries.keys())
    dims = [0, 1, 2]
    colors = ["#89b4fa", "#a6e3a1", "#fab387"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.25

    for i, dim in enumerate(dims):
        bettis = [summaries[l].betti_numbers.get(dim, 0) for l in labels]
        ax.bar(x + i * width, bettis, width,
               label=f"β{dim}", color=colors[i], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Betti number (above median persistence)")
    ax.set_title("Betti Numbers by FGOS Direction")
    ax.legend()
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_entropy_comparison(
    summaries: Dict[str, TopologicalSummary],
    out_path: Path,
) -> str:
    """Compare persistence entropy across subsets."""
    labels = list(summaries.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, dim in enumerate([0, 1]):
        ax = axes[i]
        entropies = [summaries[l].persistence_entropy.get(dim, 0) for l in labels]
        bars = ax.barh(range(len(labels)), entropies,
                       color="#89b4fa" if dim == 0 else "#a6e3a1", alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Persistence entropy (normalized)")
        ax.set_title(f"H{dim} Entropy")
        ax.grid(True, alpha=0.15, axis="x")

    fig.suptitle("Persistence Entropy by FGOS Direction", fontsize=13)
    fig.tight_layout()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_html_report(
    global_summary: TopologicalSummary,
    per_fgos: Dict[str, TopologicalSummary],
    market_summaries: Dict[str, TopologicalSummary],
    figures_b64: Dict[str, str],
    out_path: Path,
):
    """Generate self-contained HTML report."""
    def img(key, alt=""):
        b64 = figures_b64.get(key, "")
        if not b64:
            return "<p><em>No figure</em></p>"
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;border-radius:8px;margin:10px 0;">'

    gs = global_summary

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TDA — Curriculum Topology Analysis</title>
<style>
body {{ background:#1e1e2e; color:#cdd6f4; font-family:'Inter',system-ui,sans-serif; max-width:1100px; margin:0 auto; padding:20px; }}
h1 {{ color:#89b4fa; border-bottom:2px solid #313244; padding-bottom:10px; }}
h2 {{ color:#a6e3a1; margin-top:30px; }}
h3 {{ color:#fab387; }}
.card {{ background:#313244; border-radius:12px; padding:20px; margin:15px 0; }}
.metric {{ display:inline-block; background:#45475a; border-radius:8px; padding:12px 20px; margin:5px; text-align:center; }}
.metric .value {{ font-size:24px; font-weight:bold; color:#89b4fa; }}
.metric .label {{ font-size:12px; color:#a6adc8; }}
table {{ border-collapse:collapse; width:100%; margin:10px 0; }}
th,td {{ border:1px solid #45475a; padding:8px 12px; text-align:left; }}
th {{ background:#45475a; color:#89b4fa; }}
tr:nth-child(even) {{ background:#313244; }}
</style>
</head>
<body>
<h1>Topological Data Analysis — Curriculum Embedding Spaces</h1>

<div class="card">
<div class="metric"><div class="value">{gs.n_points}</div><div class="label">Courses</div></div>
<div class="metric"><div class="value">{gs.embedding_dim}→{gs.reduced_dim or gs.embedding_dim}d</div><div class="label">Dimensions</div></div>
<div class="metric"><div class="value">{gs.betti_numbers.get(0, 0)}</div><div class="label">β₀ (components)</div></div>
<div class="metric"><div class="value">{gs.betti_numbers.get(1, 0)}</div><div class="label">β₁ (loops)</div></div>
<div class="metric"><div class="value">{gs.betti_numbers.get(2, 0)}</div><div class="label">β₂ (voids)</div></div>
<div class="metric"><div class="value">{gs.total_features}</div><div class="label">Total features</div></div>
</div>

<h2>Global Persistence Diagram</h2>
{img('global_diagram', 'Persistence diagram')}

<h2>Global Barcode</h2>
{img('global_barcode', 'Persistence barcode')}

<h2>Betti Numbers by FGOS</h2>
{img('betti_comparison', 'Betti comparison')}

<h2>Persistence Entropy by FGOS</h2>
{img('entropy_comparison', 'Entropy comparison')}

<h2>Per-FGOS Persistence Diagrams</h2>
"""

    for fgos, summary in sorted(per_fgos.items()):
        key = f"fgos_{fgos}_diagram"
        html += f"""<h3>FGOS {fgos} ({summary.n_points} courses)</h3>
<div class="card">
<p>β₀={summary.betti_numbers.get(0,0)}, β₁={summary.betti_numbers.get(1,0)}, β₂={summary.betti_numbers.get(2,0)} |
Entropy: H₀={summary.persistence_entropy.get(0,0):.3f}, H₁={summary.persistence_entropy.get(1,0):.3f}</p>
{img(key, f'FGOS {fgos}')}
</div>
"""

    if market_summaries:
        html += "<h2>Market Space Topology</h2>"
        for name, summary in market_summaries.items():
            key = f"market_{name}_diagram"
            html += f"""<h3>{name} ({summary.n_points} items)</h3>
<div class="card">
<p>β₀={summary.betti_numbers.get(0,0)}, β₁={summary.betti_numbers.get(1,0)}, β₂={summary.betti_numbers.get(2,0)} |
Entropy: H₀={summary.persistence_entropy.get(0,0):.3f}, H₁={summary.persistence_entropy.get(1,0):.3f}</p>
{img(key, name)}
</div>
"""

    html += """
<hr>
<p style="color:#a6adc8;font-size:12px;">
Generated by ADS TDA Pipeline | ripser + persim + UMAP
</p>
</body></html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"  HTML report: {out_path}")


def main():
    print("=" * 60)
    print("Phase 2.1: Topological Data Analysis (Persistent Homology)")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_b64: Dict[str, str] = {}

    # 1. Load MISIS courses
    print("\n[1/5] Loading MISIS course data...")
    df = pd.read_csv("data/raw/misis/parsed/all_courses.csv")
    # Filter to courses with names
    df = df[df["course_name"].notna() & (df["course_name"].str.strip() != "")]
    print(f"  Courses with names: {len(df)}")

    # Build text for embedding
    texts = []
    fgos_labels = []
    for _, row in df.iterrows():
        parts = [str(row["course_name"])]
        if pd.notna(row.get("content_summary")):
            parts.append(str(row["content_summary"])[:200])
        if pd.notna(row.get("goals")):
            parts.append(str(row["goals"])[:200])
        texts.append(". ".join(parts))
        fgos_labels.append(str(row.get("fgos_code", "unknown")))

    # 2. Encode
    print("\n[2/5] Encoding texts...")
    embeddings = encode_texts(texts)
    print(f"  Embeddings shape: {embeddings.shape}")

    # 3. Global TDA (subsample if too many points — ripser O(n³) memory)
    MAX_GLOBAL = 500
    print(f"\n[3/5] Computing global persistent homology (PCA→10d, max {MAX_GLOBAL} pts)...")
    if embeddings.shape[0] > MAX_GLOBAL:
        rng = np.random.RandomState(42)
        idx = rng.choice(embeddings.shape[0], MAX_GLOBAL, replace=False)
        global_emb = embeddings[idx]
        print(f"  Subsampled: {embeddings.shape[0]} → {MAX_GLOBAL} points")
    else:
        global_emb = embeddings
    global_summary = compute_persistence(
        global_emb, max_dim=1, reduce_dim=10,
        reduce_method="pca", seed=42,
    )
    print(f"  Total features: {global_summary.total_features}")
    print(f"  Betti numbers: {global_summary.betti_numbers}")
    print(f"  Persistence entropy: {global_summary.persistence_entropy}")

    # Plot global
    figures_b64["global_diagram"] = plot_persistence_diagram(
        global_summary, "MISIS Global — Persistence Diagram",
        OUT_DIR / "persistence_diagram_global.png",
    )
    figures_b64["global_barcode"] = plot_persistence_barcode(
        global_summary, "MISIS Global — Persistence Barcode",
        OUT_DIR / "persistence_barcode_global.png",
    )

    # 4. Per-FGOS TDA
    print("\n[4/5] Computing per-FGOS topology...")
    fgos_arr = np.array(fgos_labels)
    fgos_counts = pd.Series(fgos_labels).value_counts()
    # Only analyze FGOS with >= 20 courses
    eligible_fgos = fgos_counts[fgos_counts >= 20].index.tolist()
    print(f"  FGOS with 20+ courses: {len(eligible_fgos)}")

    per_fgos: Dict[str, TopologicalSummary] = {}
    for fgos in eligible_fgos:
        mask = fgos_arr == fgos
        subset = embeddings[mask]
        # Subsample large FGOS groups
        if subset.shape[0] > 300:
            rng_fgos = np.random.RandomState(42)
            idx_f = rng_fgos.choice(subset.shape[0], 300, replace=False)
            subset = subset[idx_f]
        try:
            summary = compute_persistence(
                subset, max_dim=1, reduce_dim=min(10, max(2, subset.shape[0] // 3)),
                reduce_method="pca", seed=42,
            )
            per_fgos[fgos] = summary
            print(f"  {fgos} ({subset.shape[0]} pts): "
                  f"β₀={summary.betti_numbers.get(0,0)}, "
                  f"β₁={summary.betti_numbers.get(1,0)}, "
                  f"β₂={summary.betti_numbers.get(2,0)}")

            figures_b64[f"fgos_{fgos}_diagram"] = plot_persistence_diagram(
                summary, f"FGOS {fgos} ({subset.shape[0]} courses)",
                OUT_DIR / f"persistence_fgos_{fgos}.png",
            )
        except Exception as e:
            print(f"  {fgos}: SKIPPED ({e})")

    # Betti comparison
    if per_fgos:
        figures_b64["betti_comparison"] = plot_betti_comparison(
            per_fgos, OUT_DIR / "betti_comparison.png",
        )
        figures_b64["entropy_comparison"] = plot_entropy_comparison(
            per_fgos, OUT_DIR / "entropy_comparison.png",
        )

    # 5. Market space TDA
    print("\n[5/5] Computing market space topology...")
    market_summaries: Dict[str, TopologicalSummary] = {}
    market_files = {
        "HeadHunter": "data/raw/headhunter/artifacts.jsonl",
        "ProfStandart": "data/raw/profstandart/artifacts.jsonl",
    }

    for name, path in market_files.items():
        if not Path(path).exists():
            print(f"  {name}: file not found, skipping")
            continue

        market_texts = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                market_texts.append(obj.get("text", "")[:300])

        if len(market_texts) < 20:
            print(f"  {name}: too few items ({len(market_texts)})")
            continue

        if len(market_texts) > 500:
            market_texts = market_texts[:500]
        print(f"  {name}: encoding {len(market_texts)} texts...")
        market_emb = encode_texts(market_texts)

        try:
            msummary = compute_persistence(
                market_emb, max_dim=1, reduce_dim=15,
                reduce_method="pca", seed=42,
            )
            market_summaries[name] = msummary
            print(f"  {name}: β₀={msummary.betti_numbers.get(0,0)}, "
                  f"β₁={msummary.betti_numbers.get(1,0)}, "
                  f"β₂={msummary.betti_numbers.get(2,0)}")

            figures_b64[f"market_{name}_diagram"] = plot_persistence_diagram(
                msummary, f"{name} ({msummary.n_points} items)",
                OUT_DIR / f"persistence_market_{name}.png",
            )
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    # Generate report
    print("\n  Generating report...")
    generate_html_report(
        global_summary, per_fgos, market_summaries,
        figures_b64, OUT_DIR / "TDA_REPORT.html",
    )

    # JSON summary
    summary_data = {
        "generated": pd.Timestamp.now().isoformat(),
        "global": global_summary.to_dict(),
        "per_fgos": {k: v.to_dict() for k, v in per_fgos.items()},
        "market": {k: v.to_dict() for k, v in market_summaries.items()},
    }
    summary_path = OUT_DIR / "tda_summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2, default=str))
    print(f"  Summary: {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Global: {global_summary.n_points} courses, "
          f"β₀={global_summary.betti_numbers.get(0,0)}, "
          f"β₁={global_summary.betti_numbers.get(1,0)}, "
          f"β₂={global_summary.betti_numbers.get(2,0)}")
    print(f"Per-FGOS: {len(per_fgos)} directions analyzed")
    print(f"Market spaces: {len(market_summaries)} analyzed")
    print(f"Report: {OUT_DIR / 'TDA_REPORT.html'}")


if __name__ == "__main__":
    main()
