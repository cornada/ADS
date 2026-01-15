"""Pareto report generation for paper-ready artifacts.

Generates:
- CSV tables with objective scores
- Scatter plots showing trade-offs
- Radar charts for multi-objective visualization
- Summary statistics
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_pareto_report(run_dir: Path) -> Dict[str, Path]:
    """Build paper-friendly artifacts from experiment results.

    Generates:
    - pareto_table.csv: All Pareto-optimal options with objective scores
    - pareto_scatter.png: 2D scatter plot (first two objectives)

    Args:
        run_dir: Path to experiment run directory

    Returns:
        Dict mapping artifact name to path
    """
    pareto_path = run_dir / "pareto.json"
    results_path = run_dir / "results.json"

    pareto_data = json.loads(pareto_path.read_text(encoding="utf-8"))
    keys: List[str] = pareto_data["keys"]
    pareto: List[dict] = pareto_data["pareto"]

    # Also load full results for additional reporting
    results_data = json.loads(results_path.read_text(encoding="utf-8"))

    out_dir = run_dir / "paper_artifacts"
    figs_dir = out_dir / "figures"
    tabs_dir = out_dir / "paper_tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tabs_dir.mkdir(parents=True, exist_ok=True)

    # Pareto table
    rows = []
    for p in pareto:
        row = {"option_id": p["option_id"], "feasible": p["feasible"]}
        for k in keys:
            row[k] = p["objectives"][k]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(keys) > 0:
        df = df.sort_values(by=keys[0], ascending=False)  # Higher is better
    table_path = tabs_dir / "pareto_table.csv"
    df.to_csv(table_path, index=False)

    # Full results table
    all_rows = []
    for r in results_data.get("results", []):
        row = {"option_id": r["option_id"], "feasible": r["feasible"]}
        for k in keys:
            row[k] = r["objectives"].get(k)
        all_rows.append(row)
    all_df = pd.DataFrame(all_rows)
    all_table_path = tabs_dir / "all_results.csv"
    all_df.to_csv(all_table_path, index=False)

    # Scatter plot (first two objectives)
    fig_path = figs_dir / "pareto_scatter.png"
    if len(keys) >= 2 and len(df) > 0:
        xk, yk = keys[0], keys[1]
        xs = df[xk].to_list()
        ys = df[yk].to_list()

        plt.figure(figsize=(8, 6))
        plt.scatter(xs, ys, s=100, alpha=0.7, edgecolors='black', linewidth=1)

        # Annotate points
        for i, (x, y) in enumerate(zip(xs, ys)):
            option_id = df.iloc[i]["option_id"]
            # Shorten option_id for display
            label = option_id.split(":")[-1] if ":" in option_id else option_id
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

        plt.xlabel(f"{xk} (higher is better)", fontsize=12)
        plt.ylabel(f"{yk} (higher is better)", fontsize=12)
        plt.title("Pareto-Optimal Options", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
    else:
        # Create empty placeholder
        fig_path.write_bytes(b"")

    # Summary statistics
    summary = build_summary(pareto, keys, results_data)
    summary_path = tabs_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Radar chart for Pareto options
    radar_path = figs_dir / "pareto_radar.png"
    if len(keys) >= 3 and len(pareto) > 0:
        build_radar_chart(pareto, keys, radar_path)
    else:
        radar_path.write_bytes(b"")

    return {
        "table": table_path,
        "figure": fig_path,
        "radar": radar_path,
        "all_results": all_table_path,
        "summary": summary_path,
    }


def build_summary(
    pareto: List[dict],
    keys: List[str],
    results_data: dict,
) -> Dict[str, Any]:
    """Build summary statistics for the experiment.

    Args:
        pareto: List of Pareto-optimal results
        keys: Objective names
        results_data: Full results dict

    Returns:
        Summary statistics dict
    """
    all_results = results_data.get("results", [])

    # Compute stats per objective
    obj_stats = {}
    for k in keys:
        values = [r["objectives"].get(k) for r in all_results if k in r.get("objectives", {})]
        if values:
            import numpy as np
            obj_stats[k] = {
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
            }

    # Identify best option per objective
    best_per_objective = {}
    for k in keys:
        best = max(all_results, key=lambda r: r["objectives"].get(k, -float("inf")))
        best_per_objective[k] = {
            "option_id": best["option_id"],
            "score": round(best["objectives"].get(k, 0), 4),
        }

    return {
        "encoder": results_data.get("encoder", "unknown"),
        "num_options": len(all_results),
        "num_pareto": len(pareto),
        "pareto_ratio": round(len(pareto) / max(len(all_results), 1), 4),
        "objectives": keys,
        "objective_stats": obj_stats,
        "best_per_objective": best_per_objective,
        "feasible_count": sum(1 for r in all_results if r.get("feasible", True)),
    }


def export_results_csv(
    results: List[dict],
    output_path: Path,
    objective_keys: Optional[List[str]] = None,
) -> None:
    """Export results to CSV file.

    Args:
        results: List of result dicts with option_id, objectives, feasible
        output_path: Path to write CSV
        objective_keys: Column order for objectives (auto-detect if None)
    """
    if not results:
        output_path.write_text("option_id,feasible\n", encoding="utf-8")
        return

    if objective_keys is None:
        # Auto-detect from first result
        objective_keys = sorted(results[0].get("objectives", {}).keys())

    fieldnames = ["option_id", "feasible"] + objective_keys

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "option_id": r["option_id"],
                "feasible": r.get("feasible", True),
            }
            for k in objective_keys:
                row[k] = r.get("objectives", {}).get(k)
            writer.writerow(row)


def build_radar_chart(
    pareto: List[dict],
    keys: List[str],
    output_path: Path,
    title: str = "Pareto-Optimal Options",
) -> None:
    """Build a radar chart showing Pareto-optimal options across all objectives.

    Args:
        pareto: List of Pareto-optimal solutions
        keys: Objective names
        output_path: Path to save the figure
        title: Chart title
    """
    if not pareto or len(keys) < 3:
        return

    # Number of objectives
    n_obj = len(keys)

    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, n_obj, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(pareto)))

    # Normalize values to [0, 1] for better visualization
    all_values = {k: [] for k in keys}
    for p in pareto:
        for k in keys:
            all_values[k].append(p["objectives"].get(k, 0))

    min_vals = {k: min(v) if v else 0 for k, v in all_values.items()}
    max_vals = {k: max(v) if v else 1 for k, v in all_values.items()}

    # Plot each Pareto option
    for i, p in enumerate(pareto):
        values = []
        for k in keys:
            v = p["objectives"].get(k, 0)
            # Normalize to [0, 1]
            range_val = max_vals[k] - min_vals[k]
            if range_val > 1e-8:
                norm_v = (v - min_vals[k]) / range_val
            else:
                norm_v = 0.5
            values.append(norm_v)

        values += values[:1]  # Close the polygon

        # Get label
        option_id = p["option_id"]
        label = option_id.split(":")[-1] if ":" in option_id else option_id

        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys, fontsize=10)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.title(title, fontsize=14, y=1.08)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
