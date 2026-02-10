#!/usr/bin/env python3
"""Prerequisite DAG analysis on MISIS curriculum data.

Builds directed acyclic graphs from course prerequisites and computes:
- DAG statistics per FGOS direction
- Critical paths (longest prerequisite chains)
- Bottleneck courses (highest betweenness centrality)
- Topological layers (semester ordering)
- DAG vs embedding distance comparison (hidden connections)
- Figures and HTML report

Usage:
    python -m experiments.scripts.run_dag_analysis
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from ads_core.graph.prerequisite_dag import (
    build_dag,
    ensure_dag,
    compute_dag_stats,
    find_bottlenecks,
    find_critical_paths,
    topological_layers,
    compare_dag_vs_embedding,
    per_fgos_dags,
)

PREREQ_CSV = ROOT / "data" / "raw" / "misis" / "prerequisites.csv"
PARSED_2024 = ROOT / "data" / "raw" / "misis" / "parsed" / "2024_courses.csv"
REPORT_DIR = ROOT / "experiments" / "reports" / "dag_analysis"
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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_prerequisites() -> pd.DataFrame:
    df = pd.read_csv(PREREQ_CSV)
    # Remove self-loops
    df = df[df["course_name"] != df["prerequisite_name"]]
    return df


def build_course_embeddings(df_prereqs: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Encode course names appearing in the DAG."""
    from sentence_transformers import SentenceTransformer

    names_raw = set(df_prereqs["course_name"].dropna()) | set(df_prereqs["prerequisite_name"].dropna())
    all_names = sorted(str(n) for n in names_raw if isinstance(n, str))
    print(f"  Encoding {len(all_names)} unique node names...")
    model = SentenceTransformer(ENCODER_NAME)
    embs = model.encode(all_names, show_progress_bar=True, batch_size=128)
    return {name: embs[i] for i, name in enumerate(all_names)}


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_global_dag_layout(G: nx.DiGraph, stats, out_path: Path):
    """Full DAG visualization (spring layout, colored by layer)."""
    set_style()
    fig, ax = plt.subplots(figsize=(16, 12))

    # Compute layers for coloring
    dag = ensure_dag(G)
    layers = topological_layers(dag)
    node_layer = {}
    for i, layer in enumerate(layers):
        for node in layer:
            node_layer[node] = i
    max_layer = max(node_layer.values()) if node_layer else 1

    # Layout
    pos = nx.spring_layout(dag, k=0.5, iterations=50, seed=42)

    # Color by layer
    colors = [plt.cm.viridis(node_layer.get(n, 0) / max(max_layer, 1)) for n in dag.nodes()]

    nx.draw_networkx_edges(dag, pos, ax=ax, alpha=0.1, edge_color="#30363d",
                           arrows=True, arrowsize=5, width=0.3)
    nx.draw_networkx_nodes(dag, pos, ax=ax, node_color=colors, node_size=15, alpha=0.7)

    ax.set_title(
        f"MISIS Prerequisite DAG (2024)\n"
        f"{stats.n_nodes} nodes, {stats.n_edges} edges, "
        f"{stats.n_components} components, depth={stats.longest_path_length}",
        fontsize=14, fontweight="bold",
    )

    # Legend
    for i in range(0, min(max_layer + 1, 6)):
        ax.scatter([], [], c=[plt.cm.viridis(i / max(max_layer, 1))],
                   label=f"Layer {i}", s=30)
    if max_layer > 5:
        ax.scatter([], [], c=[plt.cm.viridis(1.0)], label=f"Layer {max_layer}", s=30)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.5)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_fgos_dag_stats(fgos_stats: Dict[str, Dict], out_path: Path):
    """Bar chart of DAG stats per FGOS direction."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    codes = sorted(fgos_stats.keys(), key=lambda c: -fgos_stats[c]["n_edges"])
    codes = codes[:20]  # Top 20

    vals = {k: [fgos_stats[c].get(k, 0) for c in codes] for k in
            ["n_nodes", "n_edges", "longest_path_length", "n_components"]}
    labels = [c[:10] for c in codes]

    for ax, (key, title, color) in zip(axes.flat, [
        ("n_nodes", "Nodes (courses)", "#58a6ff"),
        ("n_edges", "Edges (prerequisites)", "#3fb950"),
        ("longest_path_length", "Longest Chain (depth)", "#d2a8ff"),
        ("n_components", "Components", "#f0883e"),
    ]):
        ax.barh(range(len(codes)), vals[key], color=color, edgecolor="#0d1117")
        ax.set_yticks(range(len(codes)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Prerequisite DAG Statistics per FGOS Direction", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_bottleneck_chart(bottlenecks: List[Dict], out_path: Path):
    """Horizontal bar chart of top bottleneck courses."""
    set_style()
    if not bottlenecks:
        return

    fig, ax = plt.subplots(figsize=(14, max(6, len(bottlenecks) * 0.5)))

    names = [b["node"][:45] for b in reversed(bottlenecks)]
    btwn = [b["betweenness"] for b in reversed(bottlenecks)]
    descs = [b["descendants"] for b in reversed(bottlenecks)]

    y_pos = range(len(names))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(y_pos, btwn, color=colors, edgecolor="#0d1117")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Betweenness Centrality", fontsize=12)
    ax.set_title("Top Bottleneck Courses\n(removing these would break the most learning paths)",
                 fontsize=14, fontweight="bold")

    for bar, desc in zip(bars, reversed(descs)):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"→{desc} courses", va="center", fontsize=7, color="#8b949e")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_critical_path_visualization(G: nx.DiGraph, critical_paths, out_path: Path):
    """Visualize top 3 critical paths highlighted on the DAG."""
    set_style()
    if not critical_paths:
        return

    fig, ax = plt.subplots(figsize=(18, 6))

    # Show top 3 paths as horizontal chains
    colors = ["#58a6ff", "#3fb950", "#d2a8ff"]
    y_offset = 0
    for i, cp in enumerate(critical_paths[:3]):
        path = cp["path"]
        for j, node in enumerate(path):
            color = colors[i]
            ax.add_patch(plt.Rectangle(
                (j * 2, y_offset - 0.4), 1.8, 0.8,
                facecolor=color, alpha=0.3, edgecolor=color, linewidth=1.5,
            ))
            label = node[:20] + ("..." if len(node) > 20 else "")
            ax.text(j * 2 + 0.9, y_offset, label,
                    ha="center", va="center", fontsize=6, color="#c9d1d9")
            if j < len(path) - 1:
                ax.annotate("", xy=((j + 1) * 2, y_offset), xytext=(j * 2 + 1.8, y_offset),
                            arrowprops=dict(arrowstyle="->", color=color, lw=2))
        fgos = cp.get("fgos_code", "?")
        ax.text(-0.5, y_offset, f"[{fgos}]\nlen={cp['length']}",
                ha="right", va="center", fontsize=8, color=colors[i], fontweight="bold")
        y_offset -= 1.5

    ax.set_xlim(-2, max(len(cp["path"]) for cp in critical_paths[:3]) * 2 + 1)
    ax.set_ylim(y_offset - 1, 1)
    ax.set_title("Top 3 Critical Paths (longest prerequisite chains)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_layer_distribution(layers: List[List[str]], out_path: Path):
    """Histogram of courses per topological layer (semester proxy)."""
    set_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    layer_sizes = [len(layer) for layer in layers]
    x = range(len(layer_sizes))

    ax.bar(x, layer_sizes, color="#58a6ff", edgecolor="#0d1117")
    ax.set_xlabel("Topological Layer (semester proxy)", fontsize=12)
    ax.set_ylabel("Number of courses", fontsize=12)
    ax.set_title(
        f"Course Distribution Across Topological Layers\n"
        f"({len(layers)} layers, {sum(layer_sizes)} courses)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_dag_vs_embedding(dag_dists, emb_dists, out_path: Path):
    """Scatter plot: DAG distance vs embedding distance."""
    set_style()

    # Filter out unreachable pairs (dag_dist == -1)
    mask = dag_dists >= 0
    dd = dag_dists[mask]
    ed = emb_dists[mask]

    if len(dd) < 10:
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(dd + np.random.uniform(-0.2, 0.2, len(dd)),  # jitter
               ed, s=3, alpha=0.3, c="#58a6ff")

    # Add trend line
    if len(dd) > 50:
        z = np.polyfit(dd, ed, 1)
        p = np.poly1d(z)
        x_line = np.linspace(dd.min(), dd.max(), 100)
        ax.plot(x_line, p(x_line), "--", color="#f0883e", lw=2,
                label=f"trend: {z[0]:.4f}x + {z[1]:.3f}")

    corr = np.corrcoef(dd, ed)[0, 1]

    ax.set_xlabel("DAG distance (hops)", fontsize=12)
    ax.set_ylabel("Embedding distance (cosine)", fontsize=12)
    ax.set_title(
        f"DAG Distance vs Embedding Distance\n"
        f"Pearson r = {corr:.3f} (n={len(dd)} pairs)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_single_fgos_dag(G: nx.DiGraph, fgos: str, stats, out_path: Path):
    """Detailed DAG visualization for a single FGOS direction."""
    set_style()
    fig, ax = plt.subplots(figsize=(16, 12))

    dag = ensure_dag(G)
    layers = topological_layers(dag)
    node_layer = {}
    for i, layer in enumerate(layers):
        for node in layer:
            node_layer[node] = i
    max_layer = max(node_layer.values()) if node_layer else 1

    # Hierarchical layout
    pos = {}
    for i, layer in enumerate(layers):
        for j, node in enumerate(layer):
            pos[node] = (j - len(layer) / 2, -i)

    colors = [plt.cm.viridis(node_layer.get(n, 0) / max(max_layer, 1)) for n in dag.nodes()]

    nx.draw_networkx_edges(dag, pos, ax=ax, alpha=0.3, edge_color="#30363d",
                           arrows=True, arrowsize=8, width=0.5)
    nx.draw_networkx_nodes(dag, pos, ax=ax, node_color=colors, node_size=50, alpha=0.8)

    # Label nodes (abbreviated)
    labels = {n: n[:15] for n in dag.nodes()}
    nx.draw_networkx_labels(dag, pos, labels, ax=ax, font_size=5, font_color="#c9d1d9")

    ax.set_title(
        f"FGOS {fgos}\n"
        f"{stats.n_nodes} courses, {stats.n_edges} prereqs, "
        f"depth={stats.longest_path_length}, components={stats.n_components}",
        fontsize=14, fontweight="bold",
    )
    ax.axis("off")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── HTML report ───────────────────────────────────────────────────────────────

def generate_html_report(
    global_stats: Dict,
    fgos_stats: Dict,
    bottlenecks: List[Dict],
    critical_paths: List[Dict],
    hidden_connections: List[Dict],
    fig_dir: Path,
    out_path: Path,
):
    def img_b64(path: Path) -> str:
        if not path.exists():
            return ""
        data = base64.b64encode(path.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" style="width:100%;max-width:1000px"/>'

    # Global stats table
    gs = global_stats
    global_html = "<table>"
    for k, v in gs.items():
        if k != "longest_path":
            global_html += f"<tr><th>{k}</th><td>{v}</td></tr>"
    global_html += "</table>"

    # Bottleneck table
    bn_html = "<table><tr><th>Course</th><th>Betweenness</th><th>In-degree</th>"
    bn_html += "<th>Out-degree</th><th>Descendants</th></tr>"
    for b in bottlenecks[:15]:
        bn_html += (f"<tr><td>{b['node'][:50]}</td><td>{b['betweenness']:.4f}</td>"
                    f"<td>{b['in_degree']}</td><td>{b['out_degree']}</td>"
                    f"<td>{b['descendants']}</td></tr>")
    bn_html += "</table>"

    # Critical paths
    cp_html = "<ol>"
    for cp in critical_paths[:5]:
        path_str = " → ".join(cp["path"][:8])
        if len(cp["path"]) > 8:
            path_str += f" → ... ({len(cp['path'])} total)"
        cp_html += f"<li><b>[{cp.get('fgos_code', '?')}]</b> len={cp['length']}: {path_str}</li>"
    cp_html += "</ol>"

    # Hidden connections
    hc_html = "<ol>"
    for hc in hidden_connections[:15]:
        hc_html += (
            f"<li><b>{hc['course_a'][:30]}</b> ↔ <b>{hc['course_b'][:30]}</b> "
            f"— DAG: {hc['dag_distance']} hops, Emb: {hc['embedding_distance']:.3f} "
            f"({hc['connection_type']})</li>"
        )
    hc_html += "</ol>"

    # FGOS summary table
    fgos_html = "<table><tr><th>FGOS</th><th>Nodes</th><th>Edges</th>"
    fgos_html += "<th>Depth</th><th>Components</th><th>Density</th></tr>"
    for code in sorted(fgos_stats.keys(), key=lambda c: -fgos_stats[c]["n_edges"]):
        s = fgos_stats[code]
        fgos_html += (f"<tr><td>{code}</td><td>{s['n_nodes']}</td><td>{s['n_edges']}</td>"
                      f"<td>{s['longest_path_length']}</td><td>{s['n_components']}</td>"
                      f"<td>{s['density']:.4f}</td></tr>")
    fgos_html += "</table>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>DAG Analysis: MISIS Prerequisite Structure</title>
<style>
  body {{ background: #0d1117; color: #c9d1d9; font-family: monospace; padding: 20px; }}
  h1 {{ color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 10px; }}
  h2 {{ color: #3fb950; margin-top: 40px; }}
  h3 {{ color: #d2a8ff; }}
  table {{ border-collapse: collapse; margin: 10px 0; }}
  th, td {{ border: 1px solid #30363d; padding: 6px 10px; text-align: center; }}
  th {{ background: #161b22; color: #58a6ff; }}
  td {{ background: #0d1117; }}
  ol {{ padding-left: 20px; }}
  li {{ margin: 4px 0; }}
  img {{ border: 1px solid #30363d; border-radius: 8px; margin: 10px 0; }}
</style>
</head>
<body>
<h1>Prerequisite DAG Analysis: MISIS Curriculum</h1>
<p>Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>

<h2>1. Global DAG Statistics</h2>
{global_html}

<h2>2. Full DAG Visualization</h2>
{img_b64(fig_dir / "F1_global_dag.png")}

<h2>3. Per-FGOS Statistics</h2>
{fgos_html}
{img_b64(fig_dir / "F2_fgos_stats.png")}

<h2>4. Bottleneck Courses</h2>
<p>Courses with highest betweenness centrality — removing them breaks the most learning paths.</p>
{bn_html}
{img_b64(fig_dir / "F3_bottlenecks.png")}

<h2>5. Critical Paths (longest prerequisite chains)</h2>
{cp_html}
{img_b64(fig_dir / "F4_critical_paths.png")}

<h2>6. Topological Layers (semester ordering)</h2>
{img_b64(fig_dir / "F5_layer_distribution.png")}

<h2>7. DAG Distance vs Embedding Distance</h2>
<p>Do topologically distant courses have distant embeddings?</p>
{img_b64(fig_dir / "F6_dag_vs_embedding.png")}

<h2>8. Hidden Connections</h2>
<p>Courses far in DAG but close in embedding = similar content, different prerequisites.</p>
{hc_html}

<h2>9. Top FGOS DAG Visualizations</h2>
{img_b64(fig_dir / "F7_fgos_21.05.04.png")}
{img_b64(fig_dir / "F7_fgos_09.03.01.png")}
{img_b64(fig_dir / "F7_fgos_38.03.01.png")}

</body>
</html>"""

    out_path.write_text(html)
    print(f"  Report: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Prerequisite DAG Analysis: MISIS Curriculum")
    print("=" * 70)
    t0 = time.time()

    fig_dir = REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading prerequisites...")
    df = load_prerequisites()
    print(f"  {len(df)} edges, years: {sorted(df['year'].unique())}")

    # Focus on 2024 (richest data)
    df_2024 = df[df["year"] == 2024]
    print(f"  2024: {len(df_2024)} edges")

    # ── 2. Build global DAG ───────────────────────────────────────────────────
    print("\n[2/7] Building global DAG (2024)...")
    G_raw = build_dag(df_2024)
    print(f"  Raw: {G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} edges")
    print(f"  Is DAG: {nx.is_directed_acyclic_graph(G_raw)}")

    G = ensure_dag(G_raw)
    removed = G_raw.number_of_edges() - G.number_of_edges()
    print(f"  After cycle removal: {G.number_of_edges()} edges ({removed} removed)")

    global_stats = compute_dag_stats(G)
    print(f"  Stats: {global_stats.n_nodes} nodes, {global_stats.n_edges} edges")
    print(f"  Roots: {global_stats.n_roots}, Leaves: {global_stats.n_leaves}")
    print(f"  Components: {global_stats.n_components}")
    print(f"  Longest path: {global_stats.longest_path_length} hops")
    print(f"  Max in-degree: {global_stats.max_in_degree} ({global_stats.max_in_degree_node[:40]})")
    print(f"  Max out-degree: {global_stats.max_out_degree} ({global_stats.max_out_degree_node[:40]})")

    # ── 3. Per-FGOS analysis ──────────────────────────────────────────────────
    print("\n[3/7] Analyzing per-FGOS DAGs...")
    fgos_dags = per_fgos_dags(df, year=2024)
    fgos_stats = {}
    for code, Gf in sorted(fgos_dags.items(), key=lambda x: -x[1].number_of_edges()):
        stats = compute_dag_stats(Gf)
        fgos_stats[code] = stats.to_dict()
        print(f"  {code}: {stats.n_nodes} nodes, {stats.n_edges} edges, depth={stats.longest_path_length}")

    # ── 4. Bottlenecks ────────────────────────────────────────────────────────
    print("\n[4/7] Finding bottleneck courses...")
    bottlenecks_raw = find_bottlenecks(G, top_k=20)
    bottlenecks = [b.to_dict() for b in bottlenecks_raw]
    for b in bottlenecks[:5]:
        print(f"  {b['node'][:40]} — BC={b['betweenness']:.4f}, "
              f"in={b['in_degree']}, out={b['out_degree']}, desc={b['descendants']}")

    # ── 5. Critical paths ─────────────────────────────────────────────────────
    print("\n[5/7] Finding critical paths...")
    all_critical_paths = []
    for code, Gf in sorted(fgos_dags.items(), key=lambda x: -x[1].number_of_edges()):
        cps = find_critical_paths(Gf, top_k=3, fgos_code=code)
        all_critical_paths.extend([cp.to_dict() for cp in cps])
    all_critical_paths.sort(key=lambda x: -x["length"])
    for cp in all_critical_paths[:5]:
        print(f"  [{cp['fgos_code']}] len={cp['length']}: "
              f"{' → '.join(cp['path'][:4])}{'...' if len(cp['path']) > 4 else ''}")

    # ── 6. Topological layers ─────────────────────────────────────────────────
    print("\n[6/7] Computing topological layers...")
    layers = topological_layers(G)
    print(f"  {len(layers)} layers, distribution: "
          f"{[len(l) for l in layers[:10]]}{'...' if len(layers) > 10 else ''}")

    # ── 7. DAG vs embedding comparison ────────────────────────────────────────
    print("\n[7/7] Comparing DAG vs embedding distance...")
    embeddings = build_course_embeddings(df_2024)
    dag_dists, emb_dists, hidden = compare_dag_vs_embedding(G, embeddings, sample_size=3000)

    reachable_mask = dag_dists >= 0
    if reachable_mask.any():
        corr = np.corrcoef(dag_dists[reachable_mask], emb_dists[reachable_mask])[0, 1]
        print(f"  Correlation (DAG dist vs emb dist): r={corr:.3f}")
    print(f"  Hidden connections found: {len(hidden)}")
    for h in hidden[:5]:
        print(f"    {h.course_a[:30]} ↔ {h.course_b[:30]}: "
              f"DAG={h.dag_distance}, Emb={h.embedding_distance:.3f} ({h.connection_type})")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig_global_dag_layout(G, global_stats, fig_dir / "F1_global_dag.png")
    fig_fgos_dag_stats(fgos_stats, fig_dir / "F2_fgos_stats.png")
    fig_bottleneck_chart(bottlenecks, fig_dir / "F3_bottlenecks.png")
    fig_critical_path_visualization(G, all_critical_paths, fig_dir / "F4_critical_paths.png")
    fig_layer_distribution(layers, fig_dir / "F5_layer_distribution.png")
    fig_dag_vs_embedding(dag_dists, emb_dists, fig_dir / "F6_dag_vs_embedding.png")

    # Top 3 FGOS detailed DAGs
    for code in ["21.05.04", "09.03.01", "38.03.01"]:
        if code in fgos_dags:
            stats_obj = compute_dag_stats(fgos_dags[code])
            fig_single_fgos_dag(fgos_dags[code], code, stats_obj,
                                fig_dir / f"F7_fgos_{code.replace('.', '_')}.png")

    # ── Save summary + report ─────────────────────────────────────────────────
    print("\nSaving summary and report...")
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "global_stats": global_stats.to_dict(),
        "n_fgos_directions": len(fgos_stats),
        "fgos_stats": fgos_stats,
        "top_bottlenecks": bottlenecks[:10],
        "top_critical_paths": all_critical_paths[:10],
        "dag_vs_embedding": {
            "n_pairs_sampled": len(dag_dists),
            "n_reachable": int(reachable_mask.sum()) if len(dag_dists) > 0 else 0,
            "correlation": float(corr) if len(dag_dists) > 0 and reachable_mask.any() else None,
            "n_hidden_connections": len(hidden),
        },
        "hidden_connections": [h.to_dict() for h in hidden[:20]],
    }

    json_path = REPORT_DIR / "dag_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"  Summary: {json_path}")

    generate_html_report(
        global_stats.to_dict(),
        fgos_stats,
        bottlenecks,
        all_critical_paths,
        [h.to_dict() for h in hidden],
        fig_dir,
        REPORT_DIR / "DAG_ANALYSIS_REPORT.html",
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Artifacts: {REPORT_DIR}")


if __name__ == "__main__":
    main()
