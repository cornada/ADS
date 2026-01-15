#!/usr/bin/env python
"""Cross-dataset outcome sanity check script.

Validates ADS predictions by comparing predicted market-fit to observed
career outcomes from UCB FDS and ASU career outcomes surveys.

Usage:
    python experiments/scripts/outcome_sanity.py --datasets ucb asu --embedding stub --lenses identity
    python experiments/scripts/outcome_sanity.py --datasets ucb asu --embedding sbert --lenses identity

Outputs:
    - experiments/reports/paper/cross_dataset_outcome_sanity.csv
    - experiments/reports/paper/figures/outcome_sanity_scatter.png
    - experiments/reports/paper/paper_tables/ablation_outcome_sanity.csv (if ablation)
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from ads_core.datasets.registry import load_dataset
from ads_core.outcomes.normalize import (
    load_normalized_outcomes,
    CanonicalOutcome,
    outcomes_to_dataframe,
)
from ads_core.market.targets import (
    MARKET_CATEGORIES,
    create_market_target_vectors,
    predict_market_distribution,
    keyword_baseline_prediction,
)
from ads_core.analysis.outcome_sanity import (
    OutcomeSanityResult,
    compute_outcome_sanity,
    compute_keyword_baseline,
    aggregate_sanity_results,
    UnitSanityResult,
)
from ads_core.embed.encoder import StubEncoder
from ads_core.embed.cache import DiskEmbeddingCache


def get_encoder(embedding_kind: str, embedding_model: Optional[str] = None):
    """Create encoder based on config."""
    if embedding_kind == "stub":
        return StubEncoder(d=64)
    elif embedding_kind == "sbert":
        from ads_core.embed.sentence_transformers_encoder import SentenceTransformersEncoder
        model_name = embedding_model or "all-MiniLM-L6-v2"
        return SentenceTransformersEncoder(model_name=model_name)
    else:
        raise ValueError(f"Unknown embedding kind: {embedding_kind}")


def build_unit_vectors_from_outcomes(
    outcomes: List[CanonicalOutcome],
    encoder_fn: Callable[[List[str]], np.ndarray],
) -> Dict[str, np.ndarray]:
    """Build unit vectors by embedding unit names.

    This is a simple approach: embed the unit name directly.
    For a more sophisticated approach, use course content.

    Args:
        outcomes: List of canonical outcomes
        encoder_fn: Encoder function

    Returns:
        Dict mapping unit to embedding vector
    """
    # Get unique units
    units = {}
    for o in outcomes:
        if o.unit not in units:
            units[o.unit] = o.unit_name

    # Embed all unit names
    unit_ids = list(units.keys())
    unit_names = list(units.values())
    embeddings = encoder_fn(unit_names)

    return {uid: embeddings[i] for i, uid in enumerate(unit_ids)}


def run_sanity_check(
    dataset_ids: List[str],
    embedding_kind: str = "stub",
    embedding_model: Optional[str] = None,
    out_dir: Path = Path("experiments/reports/paper"),
    include_ablation: bool = False,
) -> Dict[str, Any]:
    """Run outcome sanity check across datasets.

    Args:
        dataset_ids: List of dataset IDs (ucb, asu)
        embedding_kind: Embedding type (stub, sbert)
        embedding_model: Model name for sbert
        out_dir: Output directory
        include_ablation: Include ablation comparison

    Returns:
        Summary results dict
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "paper_tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Create encoder
    encoder = get_encoder(embedding_kind, embedding_model)
    cache_dir = out_dir / "cache_embeddings"
    cache = DiskEmbeddingCache(root=cache_dir, model_id=encoder.model_id)

    def encode_with_cache(texts: List[str]) -> np.ndarray:
        return cache.get_or_compute(texts, encoder.encode, verbose=True)

    # Create market target vectors
    print("\nCreating market target vectors...")
    market_texts = [cat.to_text() for cat in MARKET_CATEGORIES.values()]
    market_embeddings = encode_with_cache(market_texts)
    market_vectors = {
        cat_id: market_embeddings[i]
        for i, cat_id in enumerate(MARKET_CATEGORIES.keys())
    }

    all_results: List[OutcomeSanityResult] = []
    all_unit_results: List[Dict[str, Any]] = []

    for dataset_id in dataset_ids:
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset_id}")
        print(f"{'=' * 60}")

        try:
            # Load outcomes
            outcomes = load_normalized_outcomes(dataset_id)
            print(f"  Loaded {len(outcomes)} outcome records")

            # Build unit vectors (embed unit names)
            unit_vectors = build_unit_vectors_from_outcomes(outcomes, encode_with_cache)
            print(f"  Built vectors for {len(unit_vectors)} units")

            # Compute sanity check with embeddings
            result = compute_outcome_sanity(
                dataset_id=dataset_id,
                outcomes=outcomes,
                unit_vectors=unit_vectors,
                market_vectors=market_vectors,
                method=f"embedding_{embedding_kind}",
            )
            all_results.append(result)

            print(f"  Embedding method:")
            print(f"    Coverage: {result.coverage:.1%}")
            print(f"    Spearman (employment): {result.spearman_employment}")
            print(f"    Spearman (salary): {result.spearman_salary}")

            # Collect unit results for export
            for ur in result.unit_results:
                row = ur.to_dict()
                row["method"] = f"embedding_{embedding_kind}"
                row["dataset_id"] = dataset_id
                all_unit_results.append(row)

            # Compute keyword baseline
            baseline_result = compute_keyword_baseline(
                dataset_id=dataset_id,
                outcomes=outcomes,
                market_vectors=market_vectors,
            )
            all_results.append(baseline_result)

            print(f"  Keyword baseline:")
            print(f"    Coverage: {baseline_result.coverage:.1%}")
            print(f"    Spearman (employment): {baseline_result.spearman_employment}")
            print(f"    Spearman (salary): {baseline_result.spearman_salary}")

            for ur in baseline_result.unit_results:
                row = ur.to_dict()
                row["method"] = "keyword_baseline"
                row["dataset_id"] = dataset_id
                all_unit_results.append(row)

        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Aggregate results
    summary = aggregate_sanity_results(all_results)

    # Save main results CSV
    main_csv_path = out_dir / "cross_dataset_outcome_sanity.csv"
    _export_main_csv(all_results, main_csv_path)
    print(f"\n[OK] Main results: {main_csv_path}")

    # Save detailed unit results
    detail_csv_path = tables_dir / "outcome_sanity_details.csv"
    _export_details_csv(all_unit_results, detail_csv_path)
    print(f"[OK] Detail results: {detail_csv_path}")

    # Save ablation table
    ablation_path = tables_dir / "ablation_outcome_sanity.csv"
    _export_ablation_csv(all_results, ablation_path)
    print(f"[OK] Ablation table: {ablation_path}")

    # Generate scatter plot
    scatter_path = figures_dir / "outcome_sanity_scatter.png"
    _generate_scatter_plot(all_unit_results, scatter_path)
    print(f"[OK] Scatter plot: {scatter_path}")

    # Save summary JSON
    summary_path = out_dir / "outcome_sanity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Summary: {summary_path}")

    return summary


def _export_main_csv(results: List[OutcomeSanityResult], path: Path) -> None:
    """Export main results to CSV."""
    fieldnames = [
        "dataset_id", "method", "n_units", "n_with_outcomes", "coverage",
        "spearman_employment", "spearman_salary", "spearman_p_value",
        "mae_employment", "mae_salary_rank",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


def _export_details_csv(unit_results: List[Dict], path: Path) -> None:
    """Export detailed unit results to CSV."""
    if not unit_results:
        path.write_text("No results\n", encoding="utf-8")
        return

    fieldnames = list(unit_results[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in unit_results:
            writer.writerow(row)


def _export_ablation_csv(results: List[OutcomeSanityResult], path: Path) -> None:
    """Export ablation comparison table."""
    # Group by dataset
    by_dataset: Dict[str, Dict[str, OutcomeSanityResult]] = {}
    for r in results:
        if r.dataset_id not in by_dataset:
            by_dataset[r.dataset_id] = {}
        by_dataset[r.dataset_id][r.method] = r

    rows = []
    for dataset_id, methods in by_dataset.items():
        for method, result in methods.items():
            rows.append({
                "dataset": dataset_id,
                "method": method,
                "coverage": result.coverage,
                "spearman_emp": result.spearman_employment,
                "spearman_sal": result.spearman_salary,
                "p_value": result.spearman_p_value,
            })

    fieldnames = ["dataset", "method", "coverage", "spearman_emp", "spearman_sal", "p_value"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _generate_scatter_plot(unit_results: List[Dict], path: Path) -> None:
    """Generate scatter plot of predicted vs observed."""
    # Filter to embedding method only
    embedding_results = [r for r in unit_results if "embedding" in r.get("method", "")]

    if not embedding_results:
        # Create empty plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.savefig(path, dpi=200)
        plt.close()
        return

    # Extract data
    predicted = []
    observed_emp = []
    observed_sal = []
    labels = []
    colors = []

    color_map = {"ucb": "#1f77b4", "asu": "#ff7f0e", "mit": "#2ca02c"}

    for r in embedding_results:
        pred_score = r.get("predicted_top_score", 0)
        obs_emp = r.get("observed_employment")
        obs_sal = r.get("observed_salary")
        dataset = r.get("dataset_id", "unknown")

        if obs_emp is not None:
            predicted.append(pred_score)
            observed_emp.append(obs_emp)
            labels.append(r.get("unit_name", "")[:20])
            colors.append(color_map.get(dataset, "#7f7f7f"))

    if not predicted:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No valid data points", ha="center", va="center")
        plt.savefig(path, dpi=200)
        plt.close()
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(predicted, observed_emp, c=colors, s=80, alpha=0.7, edgecolors="black", linewidth=0.5)

    # Add labels for some points
    for i, (x, y, label) in enumerate(zip(predicted, observed_emp, labels)):
        if i < 10:  # Only label first 10 to avoid clutter
            ax.annotate(label, (x, y), fontsize=7, alpha=0.7,
                       xytext=(3, 3), textcoords="offset points")

    # Trend line
    if len(predicted) >= 3:
        z = np.polyfit(predicted, observed_emp, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(predicted), max(predicted), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5, label="Trend")

    ax.set_xlabel("Predicted Market-Fit Score", fontsize=12)
    ax.set_ylabel("Observed Employment Rate", fontsize=12)
    ax.set_title("Outcome Sanity Check: Predicted vs Observed", fontsize=14)

    # Legend for datasets
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map.get("ucb", "#1f77b4"),
               markersize=10, label="UCB"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map.get("asu", "#ff7f0e"),
               markersize=10, label="ASU"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset outcome sanity check"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ucb", "asu"],
        help="Dataset IDs to check (default: ucb asu)",
    )
    parser.add_argument(
        "--embedding",
        choices=["stub", "sbert"],
        default="stub",
        help="Embedding type (default: stub)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Model name for sbert embedding",
    )
    parser.add_argument(
        "--lenses",
        choices=["identity", "diagonal"],
        default="identity",
        help="Lens type (not used in sanity check, for compatibility)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/reports/paper"),
        help="Output directory",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Include ablation comparison",
    )

    args = parser.parse_args()

    print(f"Outcome Sanity Check")
    print(f"{'=' * 60}")
    print(f"Datasets: {args.datasets}")
    print(f"Embedding: {args.embedding}")
    print(f"Output: {args.out_dir}")

    summary = run_sanity_check(
        dataset_ids=args.datasets,
        embedding_kind=args.embedding,
        embedding_model=args.embedding_model,
        out_dir=args.out_dir,
        include_ablation=args.ablation,
    )

    print(f"\n{'=' * 60}")
    print("Sanity Check Complete!")
    print(f"{'=' * 60}")

    # Print summary
    print("\nSummary by method:")
    for method, metrics in summary.get("by_method", {}).items():
        print(f"  {method}:")
        print(f"    Mean Spearman (employment): {metrics.get('mean_spearman_employment')}")
        print(f"    Mean Spearman (salary): {metrics.get('mean_spearman_salary')}")
        print(f"    Mean coverage: {metrics.get('mean_coverage'):.1%}")


if __name__ == "__main__":
    main()
