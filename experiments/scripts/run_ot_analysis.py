#!/usr/bin/env python3
"""Optimal Transport analysis: MISIS curriculum vs Russian labor market.

Computes Wasserstein distance between:
  - MISIS 2024/2025 courses (supply)
  - HeadHunter vacancies (demand)
  - ProfStandart professional standards (demand)

Generates:
  - Transport plan visualizations
  - Surplus/deficit course identification
  - Top flow analysis (which courses serve which market needs)
  - Comparative figures for the paper

Usage:
    python -m experiments.scripts.run_ot_analysis
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from ads_core.eval.optimal_transport import (
    compute_transport,
    analyze_residuals,
    extract_top_flows,
    compare_distributions,
    cosine_cost_matrix,
    TransportResult,
    ResidualAnalysis,
)

# ── paths ─────────────────────────────────────────────────────────────────────
MISIS_PARSED_24 = ROOT / "data" / "raw" / "misis" / "parsed" / "2024_courses.csv"
MISIS_PARSED_25 = ROOT / "data" / "raw" / "misis" / "parsed" / "2025_courses.csv"
HH_ARTIFACTS = ROOT / "data" / "raw" / "headhunter" / "artifacts.jsonl"
PS_ARTIFACTS = ROOT / "data" / "raw" / "profstandart" / "artifacts.jsonl"
ESCO_ARTIFACTS = ROOT / "data" / "raw" / "esco" / "artifacts.jsonl"

REPORT_DIR = ROOT / "experiments" / "reports" / "ot_analysis"

ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ── data loading ──────────────────────────────────────────────────────────────

def load_misis_courses(year: int = 2024) -> pd.DataFrame:
    """Load MISIS parsed courses with text for embedding."""
    path = MISIS_PARSED_24 if year == 2024 else MISIS_PARSED_25
    df = pd.read_csv(path)

    # Build rich text for embedding
    texts = []
    for _, row in df.iterrows():
        parts = []
        name = str(row.get("course_name", "")).strip()
        if name and name != "nan":
            parts.append(name)
        dept = str(row.get("department", "")).strip()
        if dept and dept != "nan":
            parts.append(f"Кафедра: {dept}")
        goals = str(row.get("goals", "")).strip()
        if goals and goals != "nan":
            parts.append(goals[:500])
        elif str(row.get("content_summary", "")).strip() not in ("", "nan"):
            parts.append(str(row["content_summary"])[:500])
        comps = str(row.get("competency_codes", "")).strip()
        if comps and comps != "nan":
            parts.append(f"Компетенции: {comps}")
        texts.append(". ".join(parts) if parts else "")

    df["text"] = texts
    # Drop rows with no text
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    return df


def load_jsonl_artifacts(path: Path, max_items: int = 0) -> List[Dict[str, Any]]:
    """Load artifacts from JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            items.append(item)
            if max_items and len(items) >= max_items:
                break
    return items


def load_market_data() -> Dict[str, Tuple[List[str], List[str]]]:
    """Load all market sources. Returns {name: (texts, labels)}."""
    markets = {}

    # HeadHunter
    hh = load_jsonl_artifacts(HH_ARTIFACTS)
    hh_texts = [a["text"] for a in hh]
    hh_labels = [a["metadata"].get("name", a["artifact_id"]) for a in hh]
    markets["HeadHunter"] = (hh_texts, hh_labels)

    # ProfStandart
    ps = load_jsonl_artifacts(PS_ARTIFACTS)
    ps_texts = [a["text"] for a in ps]
    ps_labels = [a["metadata"].get("name", a["artifact_id"]) for a in ps]
    markets["ProfStandart"] = (ps_texts, ps_labels)

    # ESCO (if exists)
    if ESCO_ARTIFACTS.exists():
        esco = load_jsonl_artifacts(ESCO_ARTIFACTS)
        if esco:
            esco_texts = [a["text"] for a in esco]
            esco_labels = [a["metadata"].get("name", a["artifact_id"]) for a in esco]
            markets["ESCO"] = (esco_texts, esco_labels)

    return markets


# ── embedding ─────────────────────────────────────────────────────────────────

def encode_texts(texts: List[str], encoder_name: str = ENCODER_NAME) -> np.ndarray:
    """Encode texts using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(encoder_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)
    return np.array(embeddings, dtype=np.float32)


# ── figures ───────────────────────────────────────────────────────────────────

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


def fig_cost_matrix_heatmap(
    result: TransportResult,
    market_name: str,
    out_path: Path,
    max_show: int = 80,
):
    """Heatmap of cost matrix (sample for readability)."""
    set_style()
    C = result.cost_matrix

    # Sample for visualization
    n_src = min(max_show, C.shape[0])
    n_tgt = min(max_show, C.shape[1])
    idx_src = np.linspace(0, C.shape[0] - 1, n_src, dtype=int)
    idx_tgt = np.linspace(0, C.shape[1] - 1, n_tgt, dtype=int)
    C_sub = C[np.ix_(idx_src, idx_tgt)]

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(C_sub, aspect="auto", cmap="inferno", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine distance", fontsize=12)

    ax.set_xlabel(f"{market_name} targets (sampled)", fontsize=12)
    ax.set_ylabel("MISIS courses (sampled)", fontsize=12)
    ax.set_title(
        f"Cost Matrix: MISIS → {market_name}\n"
        f"({result.n_source} courses × {result.n_target} targets)",
        fontsize=14, fontweight="bold",
    )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_transport_plan(
    result: TransportResult,
    market_name: str,
    out_path: Path,
    max_show: int = 80,
):
    """Heatmap of the transport plan (coupling matrix)."""
    set_style()
    T = result.transport_plan

    n_src = min(max_show, T.shape[0])
    n_tgt = min(max_show, T.shape[1])
    idx_src = np.linspace(0, T.shape[0] - 1, n_src, dtype=int)
    idx_tgt = np.linspace(0, T.shape[1] - 1, n_tgt, dtype=int)
    T_sub = T[np.ix_(idx_src, idx_tgt)]

    fig, ax = plt.subplots(figsize=(14, 10))
    # Use log scale to see sparse structure
    vmin = T_sub[T_sub > 0].min() if (T_sub > 0).any() else 1e-8
    im = ax.imshow(
        T_sub + 1e-12,
        aspect="auto", cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=T_sub.max()),
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Transport mass (log scale)", fontsize=12)

    ax.set_xlabel(f"{market_name} targets (sampled)", fontsize=12)
    ax.set_ylabel("MISIS courses (sampled)", fontsize=12)
    ax.set_title(
        f"OT Plan: MISIS → {market_name}\n"
        f"W = {result.wasserstein_distance:.4f}",
        fontsize=14, fontweight="bold",
    )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_surplus_deficit(
    residuals: ResidualAnalysis,
    market_name: str,
    out_path: Path,
):
    """Dual histogram: per-source cost (surplus) and per-target coverage (deficit)."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: per-source transport cost distribution
    ax1.hist(residuals.per_source_cost, bins=50, color="#58a6ff", alpha=0.8, edgecolor="#0d1117")
    median_cost = np.median(residuals.per_source_cost)
    ax1.axvline(median_cost, color="#f0883e", ls="--", lw=2, label=f"median={median_cost:.3f}")
    ax1.set_xlabel("Average transport cost per course", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Course surplus indicator\n(high cost = poor market match)", fontsize=13)
    ax1.legend(fontsize=10)

    # Right: per-target coverage distribution
    ax2.hist(residuals.per_target_coverage, bins=50, color="#3fb950", alpha=0.8, edgecolor="#0d1117")
    median_cov = np.median(residuals.per_target_coverage)
    ax2.axvline(median_cov, color="#f0883e", ls="--", lw=2, label=f"median={median_cov:.3f}")
    ax2.set_xlabel("Demand coverage (received / expected)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title(f"Market deficit indicator ({market_name})\n(low = unmet demand)", fontsize=13)
    ax2.legend(fontsize=10)

    fig.suptitle(f"OT Residuals: MISIS → {market_name}", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_top_flows_sankey(
    flows: List[Dict[str, Any]],
    market_name: str,
    out_path: Path,
    top_k: int = 15,
):
    """Horizontal bar chart showing top source→target flows."""
    set_style()
    flows = flows[:top_k]
    if not flows:
        return

    fig, ax = plt.subplots(figsize=(14, max(6, len(flows) * 0.5)))

    labels = []
    widths = []
    colors_list = []
    for f in reversed(flows):
        src = f["source"][:35]
        tgt = f["target"][:35]
        labels.append(f"{src} → {tgt}")
        widths.append(f["flow_mass"])
        colors_list.append(f["cost"])

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, widths, color=plt.cm.coolwarm(np.array(colors_list) / max(colors_list)))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Transport mass", fontsize=12)
    ax.set_title(f"Top {top_k} Flows: MISIS → {market_name}", fontsize=14, fontweight="bold")

    # Add cost annotation
    for bar, cost in zip(bars, reversed([f["cost"] for f in flows])):
        ax.text(
            bar.get_width() + 0.0001, bar.get_y() + bar.get_height() / 2,
            f"d={cost:.2f}", va="center", fontsize=7, color="#8b949e",
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_wasserstein_comparison(
    comparisons: Dict[str, float],
    out_path: Path,
):
    """Bar chart comparing Wasserstein distances across market sources."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(comparisons.keys())
    values = list(comparisons.values())
    colors = ["#58a6ff", "#3fb950", "#d2a8ff", "#f0883e"][:len(names)]

    bars = ax.bar(names, values, color=colors, edgecolor="#0d1117", width=0.6)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", fontsize=12, fontweight="bold",
        )

    ax.set_ylabel("Wasserstein distance (cosine)", fontsize=12)
    ax.set_title(
        "MISIS Curriculum vs Market Sources\n(lower = better alignment)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_pca_transport(
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    result: TransportResult,
    market_name: str,
    out_path: Path,
    n_flows: int = 30,
):
    """PCA scatter plot with top transport flows drawn as arrows."""
    from sklearn.decomposition import PCA
    set_style()

    # Combine and PCA
    combined = np.vstack([source_emb, target_emb])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(combined)

    src_coords = coords[:len(source_emb)]
    tgt_coords = coords[len(source_emb):]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot points
    ax.scatter(src_coords[:, 0], src_coords[:, 1], c="#58a6ff", s=8, alpha=0.5, label="MISIS courses")
    ax.scatter(tgt_coords[:, 0], tgt_coords[:, 1], c="#f0883e", s=8, alpha=0.5, label=f"{market_name} targets")

    # Draw top flows as arrows
    T = result.transport_plan
    flat = T.ravel()
    top_idx = np.argsort(-flat)[:n_flows]
    rows, cols = np.unravel_index(top_idx, T.shape)

    for r, c in zip(rows, cols):
        mass = T[r, c]
        if mass < 1e-10:
            continue
        x0, y0 = src_coords[r]
        x1, y1 = tgt_coords[c]
        alpha = min(1.0, mass * T.shape[0] * T.shape[1] * 2)
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->", color="#3fb950",
                alpha=max(0.1, alpha), lw=max(0.3, alpha * 2),
            ),
        )

    ax.legend(fontsize=11)
    explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)", fontsize=12)
    ax.set_title(
        f"OT Flow: MISIS → {market_name} (top {n_flows} flows)\n"
        f"W = {result.wasserstein_distance:.4f}",
        fontsize=14, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_fgos_market_alignment(
    df_courses: pd.DataFrame,
    residuals: ResidualAnalysis,
    market_name: str,
    out_path: Path,
):
    """Box plot of transport cost grouped by FGOS direction."""
    set_style()

    df = df_courses.copy()
    df["transport_cost"] = residuals.per_source_cost

    # Group by FGOS prefix (2-digit)
    df["fgos_prefix"] = df["fgos_code"].str[:5]  # e.g., "09.03" or "01.04"
    grouped = df.groupby("fgos_prefix")["transport_cost"].agg(["mean", "count", "std"])
    grouped = grouped[grouped["count"] >= 5].sort_values("mean")

    if len(grouped) < 3:
        return

    fig, ax = plt.subplots(figsize=(14, max(6, len(grouped) * 0.4)))

    y_pos = range(len(grouped))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(grouped)))
    ax.barh(
        y_pos, grouped["mean"],
        xerr=grouped["std"], color=colors,
        edgecolor="#0d1117", capsize=3,
    )
    labels = [f"{code} (n={int(row['count'])})" for code, row in grouped.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean transport cost to market", fontsize=12)
    ax.set_title(
        f"FGOS Direction → {market_name} Alignment\n"
        "(lower = better match with labor market)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── HTML report ───────────────────────────────────────────────────────────────

def generate_html_report(
    results: Dict[str, Dict[str, Any]],
    fig_dir: Path,
    out_path: Path,
):
    """Generate self-contained dark-theme HTML report."""
    import base64

    def img_b64(path: Path) -> str:
        if not path.exists():
            return ""
        data = base64.b64encode(path.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" style="width:100%;max-width:900px"/>'

    sections = []
    for market_name, info in results.items():
        slug = market_name.lower().replace(" ", "_")
        metrics = info.get("metrics", {})
        surplus_courses = info.get("surplus_top", [])
        deficit_targets = info.get("deficit_top", [])
        top_flows = info.get("top_flows", [])

        # Metrics table
        metrics_html = "<table><tr>"
        for k, v in metrics.items():
            metrics_html += f"<th>{k}</th>"
        metrics_html += "</tr><tr>"
        for k, v in metrics.items():
            metrics_html += f"<td>{v}</td>"
        metrics_html += "</tr></table>"

        # Surplus list
        surplus_html = "<ol>"
        for s in surplus_courses[:10]:
            surplus_html += f"<li><b>{s['name']}</b> — cost: {s['cost']:.4f}</li>"
        surplus_html += "</ol>"

        # Deficit list
        deficit_html = "<ol>"
        for d in deficit_targets[:10]:
            deficit_html += f"<li><b>{d['name']}</b> — coverage: {d['coverage']:.4f}</li>"
        deficit_html += "</ol>"

        # Top flows
        flows_html = "<ol>"
        for f in top_flows[:10]:
            flows_html += (
                f"<li><b>{f['source'][:40]}</b> → <b>{f['target'][:40]}</b> "
                f"(mass: {f['flow_mass']:.5f}, dist: {f['cost']:.3f})</li>"
            )
        flows_html += "</ol>"

        sections.append(f"""
        <section>
            <h2>MISIS → {market_name}</h2>
            {metrics_html}

            <h3>PCA Transport Flow</h3>
            {img_b64(fig_dir / f"F_pca_{slug}.png")}

            <h3>Transport Plan Heatmap</h3>
            {img_b64(fig_dir / f"F_plan_{slug}.png")}

            <h3>Surplus & Deficit</h3>
            {img_b64(fig_dir / f"F_residuals_{slug}.png")}

            <h3>Top 15 Flows</h3>
            {img_b64(fig_dir / f"F_flows_{slug}.png")}

            <h3>FGOS → Market Alignment</h3>
            {img_b64(fig_dir / f"F_fgos_{slug}.png")}

            <h3>Top 10 Surplus Courses (poorest market match)</h3>
            {surplus_html}

            <h3>Top 10 Market Deficit (unmet demand)</h3>
            {deficit_html}

            <h3>Top 10 Transport Flows</h3>
            {flows_html}
        </section>
        """)

    comparison_img = img_b64(fig_dir / "F_wasserstein_comparison.png")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>OT Analysis: MISIS vs Russian Labor Market</title>
<style>
  body {{ background: #0d1117; color: #c9d1d9; font-family: monospace; padding: 20px; }}
  h1 {{ color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 10px; }}
  h2 {{ color: #3fb950; margin-top: 40px; }}
  h3 {{ color: #d2a8ff; }}
  table {{ border-collapse: collapse; margin: 10px 0; }}
  th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: center; }}
  th {{ background: #161b22; color: #58a6ff; }}
  td {{ background: #0d1117; }}
  ol {{ padding-left: 20px; }}
  li {{ margin: 4px 0; }}
  section {{ margin-bottom: 60px; }}
  img {{ border: 1px solid #30363d; border-radius: 8px; margin: 10px 0; }}
</style>
</head>
<body>
<h1>Optimal Transport: MISIS Curriculum vs Russian Labor Market</h1>
<p>Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
<p>Encoder: {ENCODER_NAME} | Regularization: Sinkhorn (eps=0.05)</p>

<h2>Wasserstein Distance Comparison</h2>
{comparison_img}

{"".join(sections)}

</body>
</html>"""

    out_path.write_text(html)
    print(f"  Report: {out_path}")


# ── main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Optimal Transport Analysis: MISIS → Russian Labor Market")
    print("=" * 70)
    t0 = time.time()

    # Create report directory
    fig_dir = REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load MISIS courses (2024 — richest data) ──────────────────────────
    print("\n[1/5] Loading MISIS courses (2024)...")
    df_courses = load_misis_courses(2024)
    print(f"  Loaded {len(df_courses)} courses with text")

    # ── 2. Load market data ───────────────────────────────────────────────────
    print("\n[2/5] Loading market data...")
    markets = load_market_data()
    for name, (texts, labels) in markets.items():
        print(f"  {name}: {len(texts)} items")

    # ── 3. Encode all texts ───────────────────────────────────────────────────
    print(f"\n[3/5] Encoding with {ENCODER_NAME}...")
    all_texts = list(df_courses["text"])
    text_offset = {"MISIS": (0, len(all_texts))}

    for name, (texts, labels) in markets.items():
        start = len(all_texts)
        all_texts.extend(texts)
        text_offset[name] = (start, start + len(texts))

    print(f"  Total texts to encode: {len(all_texts)}")
    all_emb = encode_texts(all_texts)
    print(f"  Embedding shape: {all_emb.shape}")

    misis_emb = all_emb[text_offset["MISIS"][0]:text_offset["MISIS"][1]]
    misis_labels = df_courses["course_name"].fillna("(unnamed)").tolist()

    # ── 4. Run OT for each market source ──────────────────────────────────────
    print("\n[4/5] Computing Optimal Transport...")
    all_results = {}
    wasserstein_distances = {}

    for market_name, (texts, labels) in markets.items():
        start, end = text_offset[market_name]
        market_emb = all_emb[start:end]
        slug = market_name.lower().replace(" ", "_")

        print(f"\n  --- MISIS → {market_name} ({len(misis_emb)} × {len(market_emb)}) ---")

        # Compute OT
        result = compute_transport(
            misis_emb, market_emb,
            misis_labels, labels,
            regularization=0.05,
        )
        print(f"  Wasserstein distance: {result.wasserstein_distance:.4f}")

        # Residual analysis
        residuals = analyze_residuals(result, surplus_threshold=0.2, deficit_threshold=0.2)

        # Top flows
        top_flows = extract_top_flows(result, top_k=20)

        wasserstein_distances[market_name] = result.wasserstein_distance

        # Summary metrics
        metrics = {
            "Wasserstein": f"{result.wasserstein_distance:.4f}",
            "n_courses": result.n_source,
            f"n_{market_name}": result.n_target,
            "median_cost": f"{float(np.median(result.cost_matrix)):.4f}",
            "mean_cost": f"{float(np.mean(result.cost_matrix)):.4f}",
            "surplus_courses": len(residuals.surplus_indices),
            "deficit_targets": len(residuals.deficit_indices),
        }

        # Top surplus courses
        surplus_top = [
            {"name": residuals.surplus_labels[i], "cost": residuals.surplus_scores[i]}
            for i in range(min(10, len(residuals.surplus_labels)))
        ]

        # Top deficit targets
        deficit_top = [
            {"name": residuals.deficit_labels[i], "coverage": residuals.deficit_scores[i]}
            for i in range(min(10, len(residuals.deficit_labels)))
        ]

        all_results[market_name] = {
            "metrics": metrics,
            "surplus_top": surplus_top,
            "deficit_top": deficit_top,
            "top_flows": top_flows,
        }

        # Generate figures
        print(f"  Generating figures for {market_name}...")
        fig_cost_matrix_heatmap(result, market_name, fig_dir / f"F_cost_{slug}.png")
        fig_transport_plan(result, market_name, fig_dir / f"F_plan_{slug}.png")
        fig_surplus_deficit(residuals, market_name, fig_dir / f"F_residuals_{slug}.png")
        fig_top_flows_sankey(top_flows, market_name, fig_dir / f"F_flows_{slug}.png")
        fig_pca_transport(misis_emb, market_emb, result, market_name, fig_dir / f"F_pca_{slug}.png")
        fig_fgos_market_alignment(df_courses, residuals, market_name, fig_dir / f"F_fgos_{slug}.png")

    # ── 5. Comparison figure + report ─────────────────────────────────────────
    print("\n[5/5] Generating comparison + report...")
    fig_wasserstein_comparison(wasserstein_distances, fig_dir / "F_wasserstein_comparison.png")

    # Save summary JSON
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "encoder": ENCODER_NAME,
        "n_courses": len(df_courses),
        "markets": {
            name: {
                "n_items": len(markets[name][0]),
                "wasserstein": wasserstein_distances[name],
            }
            for name in markets
        },
        "detailed_results": {
            name: {
                "metrics": info["metrics"],
                "top_surplus": info["surplus_top"][:5],
                "top_deficit": info["deficit_top"][:5],
                "top_flows": info["top_flows"][:5],
            }
            for name, info in all_results.items()
        },
    }
    json_path = REPORT_DIR / "ot_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"  Summary: {json_path}")

    # HTML report
    generate_html_report(all_results, fig_dir, REPORT_DIR / "OT_ANALYSIS_REPORT.html")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Artifacts: {REPORT_DIR}")


if __name__ == "__main__":
    main()
