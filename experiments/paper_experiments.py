#!/usr/bin/env python
"""Paper-grade experiment runner for KT9.

Runs the complete experiment matrix and generates paper-ready artifacts:
- A1: Lens ablation (identity, diagonal, learned)
- A2: Embedding ablation (stub, sbert)
- A3: Constraint sensitivity (tau = 0.1, 0.25, 0.5, 1.0)
- B1: Single-objective baseline

Outputs:
- experiments/reports/paper/paper_tables/*.csv
- experiments/reports/paper/figures/*.png
- experiments/reports/paper/summary.json

Usage:
    python -m experiments.paper_experiments
    python -m experiments.paper_experiments --quick  # stub only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    name: str
    group: str
    embedding: str = "stub"
    lenses: str = "identity"
    tau: float = 0.25
    objectives: List[str] = field(default_factory=lambda: ["market", "mission", "university", "learner"])
    seed: int = 42

    @property
    def run_id(self) -> str:
        """Generate unique run ID."""
        obj_str = "_".join(sorted(self.objectives))
        return f"paper_{self.group}_{self.name}_{self.embedding}_{self.lenses}_tau{self.tau}_{obj_str}"

    def to_hydra_args(self) -> List[str]:
        """Convert to Hydra command line arguments."""
        args = [
            f"embedding={self.embedding}",
            f"lenses={self.lenses}",
            f"constraints.autonomy_drift_tau={self.tau}",
            f"seed={self.seed}",
            f"outputs.run_id={self.run_id}",
        ]
        # Only specify objectives if not default
        if set(self.objectives) != {"market", "mission", "university", "learner"}:
            args.append(f"objectives.enabled=[{','.join(self.objectives)}]")
        return args


def get_experiment_matrix(quick: bool = False) -> List[ExperimentConfig]:
    """Build the full experiment matrix.

    Args:
        quick: If True, only run stub embedding experiments

    Returns:
        List of experiment configurations
    """
    experiments = []

    # A1: Lens ablation
    for lens in ["identity", "diagonal", "learned"]:
        experiments.append(ExperimentConfig(
            name=lens,
            group="A1_lens",
            embedding="stub",
            lenses=lens,
            tau=0.25,
        ))

    # A2: Embedding ablation
    embeddings = ["stub"] if quick else ["stub", "sbert_allminilm"]
    for emb in embeddings:
        experiments.append(ExperimentConfig(
            name=emb.replace("_", "-"),
            group="A2_embedding",
            embedding=emb,
            lenses="identity",
            tau=0.25,
        ))

    # A3: Constraint sensitivity
    for tau in [0.1, 0.25, 0.5, 1.0]:
        experiments.append(ExperimentConfig(
            name=f"tau{tau}",
            group="A3_constraint",
            embedding="stub",
            lenses="identity",
            tau=tau,
        ))

    # B1: Single-objective baseline
    experiments.append(ExperimentConfig(
        name="learner_only",
        group="B1_baseline",
        embedding="stub",
        lenses="identity",
        tau=1.0,  # No constraint
        objectives=["learner"],
    ))

    return experiments


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a single experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dict with run results or error
    """
    args = ["python", "-m", "experiments.run"] + config.to_hydra_args()

    print(f"  Running: {' '.join(args)}")

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            return {
                "status": "error",
                "config": config.run_id,
                "error": result.stderr[:500],
            }

        # Parse output to get run dir
        run_dir = None
        for line in result.stdout.split("\n"):
            if "[OK] Run dir:" in line:
                run_dir = line.split(":", 1)[1].strip()
                break

        return {
            "status": "success",
            "config": config.run_id,
            "run_dir": run_dir,
            "stdout": result.stdout,
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "config": config.run_id,
        }
    except Exception as e:
        return {
            "status": "error",
            "config": config.run_id,
            "error": str(e),
        }


def collect_results(experiments: List[ExperimentConfig], base_dir: Path) -> pd.DataFrame:
    """Collect results from all experiment runs.

    Args:
        experiments: List of experiment configs
        base_dir: Base directory for experiment reports

    Returns:
        DataFrame with aggregated results
    """
    rows = []

    for config in experiments:
        run_dir = base_dir / config.run_id
        results_path = run_dir / "results.json"
        pareto_path = run_dir / "pareto.json"

        if not results_path.exists():
            rows.append({
                "run_id": config.run_id,
                "group": config.group,
                "name": config.name,
                "embedding": config.embedding,
                "lenses": config.lenses,
                "tau": config.tau,
                "status": "missing",
            })
            continue

        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
            pareto = json.loads(pareto_path.read_text(encoding="utf-8"))

            # Extract metrics
            all_results = results.get("results", [])
            pareto_opts = pareto.get("pareto", [])

            # Compute aggregate scores
            objectives = {}
            for r in all_results:
                for k, v in r.get("objectives", {}).items():
                    if k not in objectives:
                        objectives[k] = []
                    objectives[k].append(v)

            row = {
                "run_id": config.run_id,
                "group": config.group,
                "name": config.name,
                "embedding": config.embedding,
                "lenses": config.lenses,
                "tau": config.tau,
                "status": "success",
                "n_options": len(all_results),
                "n_pareto": len(pareto_opts),
                "n_feasible": sum(1 for r in all_results if r.get("feasible", True)),
                "pareto_ratio": len(pareto_opts) / max(len(all_results), 1),
            }

            # Add mean scores per objective
            for k, values in objectives.items():
                row[f"{k}_mean"] = round(float(np.mean(values)), 4)
                row[f"{k}_std"] = round(float(np.std(values)), 4)

            rows.append(row)

        except Exception as e:
            rows.append({
                "run_id": config.run_id,
                "group": config.group,
                "status": "error",
                "error": str(e),
            })

    return pd.DataFrame(rows)


def generate_paper_tables(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """Generate paper-ready tables from results.

    Args:
        df: Results DataFrame
        output_dir: Output directory for tables

    Returns:
        Dict of table name to path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {}

    # Full results table
    full_path = output_dir / "full_results.csv"
    df.to_csv(full_path, index=False)
    tables["full"] = full_path

    # A1: Lens ablation table
    a1_df = df[df["group"] == "A1_lens"][
        ["name", "n_pareto", "n_feasible", "pareto_ratio"] +
        [c for c in df.columns if c.endswith("_mean")]
    ].copy()
    a1_path = output_dir / "table_A1_lens_ablation.csv"
    a1_df.to_csv(a1_path, index=False)
    tables["A1_lens"] = a1_path

    # A2: Embedding ablation table
    a2_df = df[df["group"] == "A2_embedding"][
        ["name", "n_pareto", "n_feasible"] +
        [c for c in df.columns if c.endswith("_mean")]
    ].copy()
    a2_path = output_dir / "table_A2_embedding_ablation.csv"
    a2_df.to_csv(a2_path, index=False)
    tables["A2_embedding"] = a2_path

    # A3: Constraint sensitivity table
    a3_df = df[df["group"] == "A3_constraint"][
        ["tau", "n_pareto", "n_feasible", "pareto_ratio"]
    ].copy()
    a3_path = output_dir / "table_A3_constraint_sensitivity.csv"
    a3_df.to_csv(a3_path, index=False)
    tables["A3_constraint"] = a3_path

    # B1: Baseline comparison
    b1_df = df[df["group"] == "B1_baseline"][
        ["name", "n_pareto"] + [c for c in df.columns if c.endswith("_mean")]
    ].copy()
    b1_path = output_dir / "table_B1_baseline.csv"
    b1_df.to_csv(b1_path, index=False)
    tables["B1_baseline"] = b1_path

    return tables


def generate_paper_figures(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """Generate paper-ready figures from results.

    Args:
        df: Results DataFrame
        output_dir: Output directory for figures

    Returns:
        Dict of figure name to path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}

    # Figure 1: Lens ablation bar chart
    a1_df = df[df["group"] == "A1_lens"]
    if len(a1_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(a1_df))
        width = 0.25

        # Get objective means
        obj_cols = [c for c in a1_df.columns if c.endswith("_mean")]
        for i, col in enumerate(obj_cols[:4]):  # Max 4 objectives
            obj_name = col.replace("_mean", "")
            values = a1_df[col].fillna(0).values
            ax.bar(x + i * width, values, width, label=obj_name)

        ax.set_xlabel("Lens Type")
        ax.set_ylabel("Mean Score")
        ax.set_title("A1: Lens Ablation - Objective Scores")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(a1_df["name"].values)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        fig_path = output_dir / "fig_A1_lens_ablation.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        figures["A1_lens"] = fig_path

    # Figure 2: Constraint sensitivity line chart
    a3_df = df[df["group"] == "A3_constraint"].sort_values("tau")
    if len(a3_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(a3_df["tau"], a3_df["n_feasible"], "o-", label="Feasible", linewidth=2, markersize=8)
        ax.plot(a3_df["tau"], a3_df["n_pareto"], "s--", label="Pareto", linewidth=2, markersize=8)

        ax.set_xlabel("Autonomy Drift Threshold (τ)")
        ax.set_ylabel("Count")
        ax.set_title("A3: Constraint Sensitivity Analysis")
        ax.legend()
        ax.grid(alpha=0.3)

        fig_path = output_dir / "fig_A3_constraint_sensitivity.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        figures["A3_constraint"] = fig_path

    # Figure 3: Pareto ratio comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = df["group"].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))

    x_offset = 0
    x_ticks = []
    x_labels = []

    for i, group in enumerate(groups):
        group_df = df[df["group"] == group]
        for j, (_, row) in enumerate(group_df.iterrows()):
            ax.bar(x_offset, row.get("pareto_ratio", 0), color=colors[i], edgecolor="black")
            x_ticks.append(x_offset)
            x_labels.append(row.get("name", "?"))
            x_offset += 1
        x_offset += 0.5  # Gap between groups

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Pareto Ratio")
    ax.set_title("Pareto Optimality Ratio Across Experiments")
    ax.grid(axis="y", alpha=0.3)

    fig_path = output_dir / "fig_pareto_ratio_comparison.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    figures["pareto_ratio"] = fig_path

    return figures


def generate_summary(
    df: pd.DataFrame,
    tables: Dict[str, Path],
    figures: Dict[str, Path],
    output_dir: Path,
) -> Path:
    """Generate summary JSON for the paper experiments.

    Args:
        df: Results DataFrame
        tables: Generated table paths
        figures: Generated figure paths
        output_dir: Output directory

    Returns:
        Path to summary JSON
    """
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_groups": {},
        "tables": {k: str(v) for k, v in tables.items()},
        "figures": {k: str(v) for k, v in figures.items()},
        "totals": {
            "experiments_run": len(df),
            "successful": len(df[df["status"] == "success"]),
            "failed": len(df[df["status"] != "success"]),
        },
    }

    # Add per-group summaries
    for group in df["group"].unique():
        group_df = df[df["group"] == group]
        summary["experiment_groups"][group] = {
            "n_experiments": len(group_df),
            "mean_pareto_ratio": round(group_df["pareto_ratio"].mean(), 4) if "pareto_ratio" in group_df else None,
            "mean_feasible": round(group_df["n_feasible"].mean(), 2) if "n_feasible" in group_df else None,
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Run paper experiments")
    parser.add_argument("--quick", action="store_true", help="Quick mode (stub only)")
    parser.add_argument("--skip-run", action="store_true", help="Skip running, only collect results")
    args = parser.parse_args()

    print("=" * 70)
    print("KT9: Paper-Grade Experiment Pack")
    print("=" * 70)

    # Setup paths
    base_dir = Path("experiments/reports")
    paper_dir = base_dir / "paper"
    tables_dir = paper_dir / "paper_tables"
    figures_dir = paper_dir / "figures"

    # Get experiment matrix
    experiments = get_experiment_matrix(quick=args.quick)
    print(f"\nExperiment matrix: {len(experiments)} experiments")
    for exp in experiments:
        print(f"  - {exp.run_id}")

    # Run experiments
    if not args.skip_run:
        print("\n" + "=" * 70)
        print("Running experiments...")
        print("=" * 70)

        for i, config in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] {config.group}/{config.name}")
            result = run_experiment(config)
            status = result.get("status", "unknown")
            if status == "success":
                print(f"  [OK] Success: {result.get('run_dir')}")
            else:
                print(f"  [FAIL] {status}: {result.get('error', '')[:100]}")

    # Collect results
    print("\n" + "=" * 70)
    print("Collecting results...")
    print("=" * 70)

    df = collect_results(experiments, base_dir)
    print(f"\nCollected {len(df)} experiment results")
    print(f"  Successful: {len(df[df['status'] == 'success'])}")
    print(f"  Failed/Missing: {len(df[df['status'] != 'success'])}")

    # Generate paper artifacts
    print("\n" + "=" * 70)
    print("Generating paper artifacts...")
    print("=" * 70)

    tables = generate_paper_tables(df, tables_dir)
    print(f"\nGenerated {len(tables)} tables:")
    for name, path in tables.items():
        print(f"  - {name}: {path}")

    figures = generate_paper_figures(df, figures_dir)
    print(f"\nGenerated {len(figures)} figures:")
    for name, path in figures.items():
        print(f"  - {name}: {path}")

    summary_path = generate_summary(df, tables, figures, paper_dir)
    print(f"\nSummary: {summary_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("PAPER ARTIFACTS READY")
    print("=" * 70)
    print(f"""
Output directory: {paper_dir}

Tables:
{chr(10).join(f'  - {p}' for p in tables.values())}

Figures:
{chr(10).join(f'  - {p}' for p in figures.values())}

Summary: {summary_path}

To regenerate: python -m experiments.paper_experiments
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
