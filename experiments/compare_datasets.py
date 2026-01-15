#!/usr/bin/env python
"""Cross-dataset comparison script for ADS experiments.

Runs the same evaluation pipeline across multiple datasets and produces
a comparison report suitable for paper tables.

Usage:
    python -m experiments.compare_datasets --datasets toy mit ucb asu
    python -m experiments.compare_datasets --datasets mit ucb --embedding sbert
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ads_core.datasets.registry import load_dataset, list_datasets
from ads_core.pipeline.dataset_pipeline import run_dataset, DatasetRunOutputs


@dataclass
class DatasetComparison:
    """Results from comparing multiple datasets."""
    datasets: List[str]
    results: Dict[str, Dict[str, Any]]
    embedding_model: str
    lens_kind: str
    timestamp: str
    summary_table: List[Dict[str, Any]] = field(default_factory=list)


def compare_datasets(
    dataset_ids: List[str],
    out_dir: Path,
    seed: int = 42,
    embedding_kind: str = "stub",
    embedding_model: Optional[str] = None,
    lens_kind: str = "identity",
    objectives: Optional[List[str]] = None,
    autonomy_tau: float = 0.3,
) -> DatasetComparison:
    """Run comparison across multiple datasets.

    Args:
        dataset_ids: List of dataset IDs to compare
        out_dir: Output directory for results
        seed: Random seed for reproducibility
        embedding_kind: Embedding type (stub, sbert)
        embedding_model: Optional model name for sbert
        lens_kind: Lens type (identity, diagonal, learned)
        objectives: List of objectives to optimize
        autonomy_tau: Autonomy drift threshold

    Returns:
        DatasetComparison with results for all datasets
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default objectives
    if objectives is None:
        objectives = ["market", "mission", "learner"]

    # Build embedding config
    if embedding_kind == "stub":
        embedding_cfg = {"kind": "stub", "d": 64}
        model_id = "stub-64"
    elif embedding_kind == "sbert":
        model_name = embedding_model or "all-MiniLM-L6-v2"
        embedding_cfg = {"kind": "sbert", "model_name": model_name}
        model_id = model_name
    else:
        raise ValueError(f"Unknown embedding kind: {embedding_kind}")

    # Build lenses config
    lenses_cfg = {"kind": lens_kind}

    results: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for dataset_id in dataset_ids:
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset_id}")
        print(f"{'=' * 60}")

        dataset_out = out_dir / dataset_id

        try:
            outputs = run_dataset(
                dataset_id=dataset_id,
                out_dir=dataset_out,
                seed=seed,
                embedding_cfg=embedding_cfg,
                lenses_cfg=lenses_cfg,
                objectives=objectives,
                autonomy_tau=autonomy_tau,
            )

            # Load detailed results
            results_data = json.loads(outputs.results_json.read_text(encoding="utf-8"))
            pareto_data = json.loads(outputs.pareto_json.read_text(encoding="utf-8"))

            # Compute summary statistics
            all_scores = results_data.get("results", [])
            objective_keys = results_data.get("objectives", [])

            # Mean scores per objective
            mean_scores = {}
            for obj in objective_keys:
                scores = [r["objectives"].get(obj, 0) for r in all_scores if r.get("feasible", True)]
                mean_scores[obj] = float(np.mean(scores)) if scores else 0.0

            # Feasibility rate
            feasible_count = sum(1 for r in all_scores if r.get("feasible", True))
            feasibility_rate = feasible_count / len(all_scores) if all_scores else 0.0

            # Pareto ratio
            pareto_ratio = outputs.pareto_count / outputs.artifact_count if outputs.artifact_count > 0 else 0.0

            results[dataset_id] = {
                "status": "success",
                "artifact_count": outputs.artifact_count,
                "pareto_count": outputs.pareto_count,
                "pareto_ratio": pareto_ratio,
                "feasibility_rate": feasibility_rate,
                "mean_scores": mean_scores,
                "objectives": objective_keys,
                "run_dir": str(outputs.run_dir),
            }

            summary_rows.append({
                "dataset": dataset_id,
                "n_artifacts": outputs.artifact_count,
                "n_pareto": outputs.pareto_count,
                "pareto_ratio": round(pareto_ratio, 3),
                "feasibility": round(feasibility_rate, 3),
                **{f"mean_{k}": round(v, 4) for k, v in mean_scores.items()},
            })

            print(f"  Artifacts: {outputs.artifact_count}")
            print(f"  Pareto: {outputs.pareto_count} ({pareto_ratio:.1%})")
            print(f"  Feasibility: {feasibility_rate:.1%}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[dataset_id] = {
                "status": "error",
                "error": str(e),
            }
            summary_rows.append({
                "dataset": dataset_id,
                "n_artifacts": 0,
                "n_pareto": 0,
                "pareto_ratio": 0,
                "feasibility": 0,
                "error": str(e),
            })

    comparison = DatasetComparison(
        datasets=dataset_ids,
        results=results,
        embedding_model=model_id,
        lens_kind=lens_kind,
        timestamp=datetime.now(timezone.utc).isoformat(),
        summary_table=summary_rows,
    )

    return comparison


def save_comparison(comparison: DatasetComparison, out_dir: Path) -> Dict[str, Path]:
    """Save comparison results to files.

    Args:
        comparison: DatasetComparison object
        out_dir: Output directory

    Returns:
        Dict mapping artifact names to paths
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full results JSON
    results_path = out_dir / "comparison_results.json"
    results_path.write_text(json.dumps({
        "datasets": comparison.datasets,
        "results": comparison.results,
        "embedding_model": comparison.embedding_model,
        "lens_kind": comparison.lens_kind,
        "timestamp": comparison.timestamp,
    }, indent=2), encoding="utf-8")

    # Save summary table (CSV-friendly)
    summary_path = out_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps({
        "columns": list(comparison.summary_table[0].keys()) if comparison.summary_table else [],
        "rows": comparison.summary_table,
    }, indent=2), encoding="utf-8")

    # Generate LaTeX table
    latex_path = out_dir / "comparison_table.tex"
    latex_content = _generate_latex_table(comparison)
    latex_path.write_text(latex_content, encoding="utf-8")

    return {
        "results_json": results_path,
        "summary_json": summary_path,
        "latex_table": latex_path,
    }


def _generate_latex_table(comparison: DatasetComparison) -> str:
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Cross-Dataset Comparison Results}",
        r"\label{tab:dataset-comparison}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & Artifacts & Pareto & Ratio & Feasibility & Mean Market \\",
        r"\midrule",
    ]

    for row in comparison.summary_table:
        if row.get("error"):
            lines.append(f"{row['dataset']} & \\multicolumn{{5}}{{c}}{{Error}} \\\\")
        else:
            mean_market = row.get("mean_market", row.get("mean_learner", 0))
            lines.append(
                f"{row['dataset']} & {row['n_artifacts']} & {row['n_pareto']} & "
                f"{row['pareto_ratio']:.3f} & {row['feasibility']:.3f} & {mean_market:.4f} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare ADS evaluation across multiple datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["toy", "mit", "ucb", "asu"],
        help="Dataset IDs to compare (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/reports/comparison"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--embedding",
        choices=["stub", "sbert"],
        default="stub",
        help="Embedding type",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Model name for sbert embedding",
    )
    parser.add_argument(
        "--lens",
        choices=["identity", "diagonal", "learned"],
        default="identity",
        help="Lens type",
    )
    parser.add_argument(
        "--autonomy-tau",
        type=float,
        default=0.3,
        help="Autonomy drift threshold",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for ds in list_datasets():
            print(f"  - {ds}")
        return

    print(f"Comparing datasets: {args.datasets}")
    print(f"Embedding: {args.embedding}")
    print(f"Lens: {args.lens}")
    print(f"Output: {args.out_dir}")

    comparison = compare_datasets(
        dataset_ids=args.datasets,
        out_dir=args.out_dir,
        seed=args.seed,
        embedding_kind=args.embedding,
        embedding_model=args.embedding_model,
        lens_kind=args.lens,
        autonomy_tau=args.autonomy_tau,
    )

    artifacts = save_comparison(comparison, args.out_dir)

    print(f"\n{'=' * 60}")
    print("Comparison complete!")
    print(f"{'=' * 60}")
    print("\nSummary:")
    for row in comparison.summary_table:
        if row.get("error"):
            print(f"  {row['dataset']}: ERROR - {row['error']}")
        else:
            print(f"  {row['dataset']}: {row['n_artifacts']} artifacts, {row['n_pareto']} pareto ({row['pareto_ratio']:.1%})")

    print("\nArtifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
