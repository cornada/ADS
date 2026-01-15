"""Sweep comparison script for ablation analysis.

Compares Pareto sets and objective statistics across multiple experiment runs.

Usage:
    python -m experiments.sweep_compare sweep_identity sweep_diagonal sweep_learned
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


def load_run(run_dir: Path) -> Dict[str, Any]:
    """Load experiment run outputs."""
    pareto_path = run_dir / "pareto.json"
    summary_path = run_dir / "paper_artifacts" / "paper_tables" / "summary.json"
    config_path = run_dir / "config_resolved.yaml"

    data = {"run_id": run_dir.name}

    if pareto_path.exists():
        data["pareto"] = json.loads(pareto_path.read_text(encoding="utf-8"))

    if summary_path.exists():
        data["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))

    if config_path.exists():
        # Just extract lens mode from config
        config_text = config_path.read_text(encoding="utf-8")
        for line in config_text.split("\n"):
            if "mode:" in line:
                data["lens_mode"] = line.split("mode:")[-1].strip()
                break

    return data


def compare_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare multiple experiment runs."""
    comparison = {
        "runs": [],
        "pareto_comparison": {},
        "objective_comparison": {},
    }

    # Collect all unique option IDs across all Pareto sets
    all_pareto_options = set()
    for run in runs:
        pareto = run.get("pareto", {}).get("pareto", [])
        for p in pareto:
            all_pareto_options.add(p["option_id"])

    # Build comparison per run
    for run in runs:
        run_id = run["run_id"]
        lens_mode = run.get("lens_mode", "unknown")
        summary = run.get("summary", {})
        pareto = run.get("pareto", {}).get("pareto", [])

        run_summary = {
            "run_id": run_id,
            "lens_mode": lens_mode,
            "num_pareto": len(pareto),
            "pareto_options": [p["option_id"] for p in pareto],
            "feasible_count": summary.get("feasible_count", 0),
            "best_per_objective": summary.get("best_per_objective", {}),
        }
        comparison["runs"].append(run_summary)

        # Track which options are Pareto for each run
        for opt_id in all_pareto_options:
            if opt_id not in comparison["pareto_comparison"]:
                comparison["pareto_comparison"][opt_id] = {}
            is_pareto = any(p["option_id"] == opt_id for p in pareto)
            comparison["pareto_comparison"][opt_id][run_id] = is_pareto

        # Objective stats per run
        obj_stats = summary.get("objective_stats", {})
        for obj_name, stats in obj_stats.items():
            if obj_name not in comparison["objective_comparison"]:
                comparison["objective_comparison"][obj_name] = {}
            comparison["objective_comparison"][obj_name][run_id] = {
                "mean": stats.get("mean"),
                "std": stats.get("std"),
            }

    return comparison


def print_comparison(comparison: Dict[str, Any]) -> None:
    """Print comparison in readable format."""
    print("=" * 60)
    print("ABLATION SWEEP COMPARISON")
    print("=" * 60)

    print("\n### Run Summary ###")
    for run in comparison["runs"]:
        print(f"\n{run['run_id']} (lens={run['lens_mode']}):")
        print(f"  Pareto options: {run['num_pareto']}")
        print(f"  Feasible: {run['feasible_count']}")
        print(f"  Options: {', '.join(run['pareto_options'])}")

    print("\n### Pareto Membership ###")
    print("Option              | " + " | ".join(
        f"{r['run_id'][:12]:>12}" for r in comparison["runs"]
    ))
    print("-" * 60)
    for opt_id, membership in comparison["pareto_comparison"].items():
        row = f"{opt_id:18} | "
        for run in comparison["runs"]:
            is_pareto = membership.get(run["run_id"], False)
            row += f"{'  PARETO  ' if is_pareto else '    -     '} | "
        print(row)

    print("\n### Objective Statistics (mean +/- std) ###")
    for obj_name, stats in comparison["objective_comparison"].items():
        print(f"\n{obj_name}:")
        for run in comparison["runs"]:
            run_stats = stats.get(run["run_id"], {})
            mean = run_stats.get("mean", 0)
            std = run_stats.get("std", 0)
            print(f"  {run['run_id']}: {mean:.4f} +/- {std:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment sweep runs")
    parser.add_argument(
        "run_ids",
        nargs="+",
        help="Run IDs to compare (directories in experiments/reports/)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("experiments/reports"),
        help="Base directory for experiment reports",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for comparison",
    )
    args = parser.parse_args()

    # Load runs
    runs = []
    for run_id in args.run_ids:
        run_dir = args.base_dir / run_id
        if not run_dir.exists():
            print(f"Warning: Run directory not found: {run_dir}")
            continue
        runs.append(load_run(run_dir))

    if not runs:
        print("No valid runs found.")
        return

    # Compare
    comparison = compare_runs(runs)

    # Output
    print_comparison(comparison)

    if args.output:
        args.output.write_text(
            json.dumps(comparison, indent=2),
            encoding="utf-8",
        )
        print(f"\nComparison saved to: {args.output}")


if __name__ == "__main__":
    main()
