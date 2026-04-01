#!/usr/bin/env python3
"""Change-point detection on MISIS temporal curriculum data.

Detects structural regime shifts across FGOS programs over 2021-2025.
Constructs 4 signals per program (n_courses, mean_credits,
competency_jaccard, course_churn) and runs change-point detection
with permutation testing for significance.

Usage:
    python -m experiments.scripts.run_changepoint_analysis

Output:
    experiments/reports/changepoint_analysis/
        changepoint_summary.json
        program_*.png  (per-program signal plots)
        consensus_heatmap.png
        aggregate_timeline.png
        CHANGEPOINT_REPORT.html
"""
from __future__ import annotations

import json
import os
import sys
import base64
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Add packages to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))

from ads_core.temporal.changepoint import (
    build_program_signals,
    detect_program_changepoints,
    permutation_test_changepoint,
    aggregate_changepoints,
    ProgramEvolution,
)


OUT_DIR = Path("experiments/reports/changepoint_analysis")

# Plot styling
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

COLORS = {
    "n_courses": "#89b4fa",
    "mean_credits": "#a6e3a1",
    "competency_jaccard": "#fab387",
    "course_churn": "#f38ba8",
}
SIGNAL_LABELS = {
    "n_courses": "Number of courses",
    "mean_credits": "Mean credits (ЗЕТ)",
    "competency_jaccard": "Competency Jaccard dist.",
    "course_churn": "Course churn rate",
}


def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_program_signals(evo: ProgramEvolution, out_path: Path) -> str:
    """Plot all signals for a single program with changepoint markers."""
    sigs = evo.signals
    if not sigs:
        return ""

    n_signals = len(sigs)
    fig, axes = plt.subplots(n_signals, 1, figsize=(10, 3 * n_signals), sharex=True)
    if n_signals == 1:
        axes = [axes]

    years = evo.years

    for ax, (sig_name, values) in zip(axes, sigs.items()):
        color = COLORS.get(sig_name, "#89b4fa")
        label = SIGNAL_LABELS.get(sig_name, sig_name)

        ax.plot(years, values, "o-", color=color, linewidth=2, markersize=8)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.2)

        # Mark changepoints
        for cp in evo.changepoints:
            if cp.signal_name == sig_name:
                ax.axvline(
                    x=(cp.year_before + cp.year_after) / 2,
                    color="#f38ba8",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                )
                ax.annotate(
                    f"Δ={cp.magnitude:.2f}",
                    xy=((cp.year_before + cp.year_after) / 2, max(values) * 0.9),
                    fontsize=8,
                    color="#f38ba8",
                    ha="center",
                )

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[-1].set_xlabel("Year")
    fig.suptitle(f"FGOS {evo.fgos_code} — Temporal Evolution", fontsize=14, y=1.02)
    fig.tight_layout()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_consensus_heatmap(
    evolutions: List[ProgramEvolution],
    out_path: Path,
) -> str:
    """Heatmap: programs × years, colored by number of changepoints."""
    from typing import List

    # Filter to programs with changepoints
    progs_with_cp = [e for e in evolutions if e.has_changepoints]
    if not progs_with_cp:
        return ""

    all_years = sorted(set(y for e in evolutions for y in e.years))
    fgos_codes = [e.fgos_code for e in progs_with_cp]

    matrix = np.zeros((len(fgos_codes), len(all_years)))
    for i, evo in enumerate(progs_with_cp):
        for cp in evo.changepoints:
            if cp.year_after in all_years:
                j = all_years.index(cp.year_after)
                matrix[i, j] += 1

    fig, ax = plt.subplots(figsize=(10, max(6, len(fgos_codes) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(all_years)))
    ax.set_xticklabels(all_years)
    ax.set_yticks(range(len(fgos_codes)))
    ax.set_yticklabels(fgos_codes, fontsize=8)
    ax.set_xlabel("Year")
    ax.set_ylabel("FGOS Code")
    ax.set_title("Change-Point Heatmap (programs × years)")

    plt.colorbar(im, ax=ax, label="# changepoints")
    fig.tight_layout()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_aggregate_timeline(
    agg: dict,
    out_path: Path,
) -> str:
    """Bar chart: total changepoints per year across all programs."""
    per_year = agg.get("per_year", {})
    if not per_year:
        return ""

    years = sorted(per_year.keys())
    counts = [per_year[y]["count"] for y in years]
    consensus = agg.get("consensus_years", {})

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        years,
        counts,
        color=["#f38ba8" if y in consensus else "#89b4fa" for y in years],
        edgecolor="#cdd6f4",
        linewidth=1.5,
    )

    # Annotate consensus years
    for y in consensus:
        if y in years:
            idx = years.index(y)
            ax.annotate(
                "CONSENSUS",
                xy=(y, counts[idx]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#f38ba8",
                fontweight="bold",
            )

    ax.set_xlabel("Year")
    ax.set_ylabel("Total changepoints detected")
    ax.set_title("Aggregate Change-Point Timeline")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_signal_distributions(
    evolutions: List[ProgramEvolution],
    out_path: Path,
) -> str:
    """Box plots of signal values per year, across all programs."""
    from typing import List

    all_years = sorted(set(y for e in evolutions for y in e.years))
    sig_names = ["n_courses", "mean_credits", "competency_jaccard", "course_churn"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, sig_name in zip(axes, sig_names):
        data_per_year = {y: [] for y in all_years}
        for evo in evolutions:
            if sig_name not in evo.signals:
                continue
            for year, val in zip(evo.years, evo.signals[sig_name]):
                data_per_year[year].append(val)

        positions = list(range(len(all_years)))
        box_data = [data_per_year[y] for y in all_years]

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=True,
        )
        color = COLORS.get(sig_name, "#89b4fa")
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set_color("#cdd6f4")

        ax.set_xticks(positions)
        ax.set_xticklabels(all_years)
        ax.set_title(SIGNAL_LABELS.get(sig_name, sig_name))
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Signal Distributions Across Programs per Year", fontsize=14, y=1.01)
    fig.tight_layout()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_permutation_results(
    perm_results: dict,
    fgos_code: str,
    signal_name: str,
    out_path: Path,
) -> str:
    """Plot permutation test null distribution vs observed changepoints."""
    fpr = perm_results.get("false_positive_rates", {})
    cps = perm_results.get("changepoints", [])

    if not fpr:
        return ""

    indices = sorted(fpr.keys())
    rates = [fpr[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(indices, rates, color="#89b4fa", alpha=0.6, label="Null distribution (FPR)")

    for cp in cps:
        color = "#a6e3a1" if cp["significant"] else "#f38ba8"
        ax.axvline(cp["index"], color=color, linestyle="--", linewidth=2)
        ax.annotate(
            f"p={cp['p_value']:.3f}",
            xy=(cp["index"], max(rates) * 0.9 if rates else 0.5),
            fontsize=9,
            color=color,
            ha="center",
        )

    ax.axhline(0.05, color="#f5c2e7", linestyle=":", alpha=0.8, label="α=0.05")
    ax.set_xlabel("Time index")
    ax.set_ylabel("False positive rate")
    ax.set_title(f"Permutation Test — {fgos_code} / {signal_name}")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_html_report(
    evolutions,
    agg,
    perm_results_map,
    figures_b64,
    out_path: Path,
):
    """Generate self-contained HTML report."""
    def img_tag(b64: str, alt: str = "") -> str:
        if not b64:
            return "<p><em>No data</em></p>"
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;border-radius:8px;margin:10px 0;">'

    cp_programs = [e for e in evolutions if e.has_changepoints]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Change-Point Detection — MISIS Temporal Analysis</title>
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
.sig {{ color:#a6e3a1; font-weight:bold; }}
.nonsig {{ color:#f38ba8; }}
</style>
</head>
<body>
<h1>Change-Point Detection — MISIS Temporal Curriculum Analysis</h1>

<div class="card">
<div class="metric"><div class="value">{agg['total_programs']}</div><div class="label">Programs analyzed</div></div>
<div class="metric"><div class="value">{agg['programs_with_changepoints']}</div><div class="label">With changepoints</div></div>
<div class="metric"><div class="value">{agg['total_changepoints']}</div><div class="label">Total changepoints</div></div>
<div class="metric"><div class="value">{len(agg.get('consensus_years', {}))}</div><div class="label">Consensus years</div></div>
</div>

<h2>Aggregate Timeline</h2>
{img_tag(figures_b64.get('aggregate_timeline', ''), 'Aggregate timeline')}

<h2>Consensus Heatmap</h2>
{img_tag(figures_b64.get('consensus_heatmap', ''), 'Consensus heatmap')}

<h2>Signal Distributions per Year</h2>
{img_tag(figures_b64.get('signal_distributions', ''), 'Signal distributions')}

<h2>Per-Year Breakdown</h2>
<table>
<tr><th>Year</th><th>Changepoints</th><th>Signals</th><th>Programs</th></tr>
"""

    for y, info in sorted(agg.get("per_year", {}).items()):
        consensus_mark = " ★" if y in agg.get("consensus_years", {}) else ""
        programs_str = ", ".join(info["programs"][:5])
        if len(info["programs"]) > 5:
            programs_str += f" (+{len(info['programs'])-5} more)"
        html += f"""<tr>
<td>{y}{consensus_mark}</td>
<td>{info['count']}</td>
<td>{', '.join(info['signals'])}</td>
<td>{programs_str}</td>
</tr>"""

    html += """</table>

<h2>Programs with Change-Points</h2>
"""

    for evo in cp_programs:
        html += f"""<h3>FGOS {evo.fgos_code} ({evo.n_years} years: {evo.years[0]}–{evo.years[-1]})</h3>
<div class="card">
"""
        fig_key = f"program_{evo.fgos_code}"
        html += img_tag(figures_b64.get(fig_key, ""), f"Signals for {evo.fgos_code}")

        html += "<table><tr><th>Signal</th><th>Year</th><th>Magnitude</th><th>Significance</th></tr>"
        for cp in evo.changepoints:
            perm_key = f"{evo.fgos_code}_{cp.signal_name}"
            perm = perm_results_map.get(perm_key, {})
            p_val = None
            for pcp in perm.get("changepoints", []):
                if pcp["index"] == cp.index:
                    p_val = pcp["p_value"]
                    break

            if p_val is not None:
                sig_class = "sig" if p_val < 0.05 else "nonsig"
                sig_text = f'<span class="{sig_class}">p={p_val:.3f}</span>'
            else:
                sig_text = "—"

            html += f"""<tr>
<td>{cp.signal_name}</td>
<td>{cp.year_before}→{cp.year_after}</td>
<td>{cp.magnitude:.3f}</td>
<td>{sig_text}</td>
</tr>"""
        html += "</table>"

        # Permutation plot if available
        for cp in evo.changepoints:
            perm_fig_key = f"perm_{evo.fgos_code}_{cp.signal_name}"
            if perm_fig_key in figures_b64:
                html += img_tag(figures_b64[perm_fig_key], f"Permutation test {evo.fgos_code}")

        html += "</div>"

    html += """
<hr>
<p style="color:#a6adc8;font-size:12px;">
Generated by ADS Change-Point Detection Pipeline | ruptures + permutation testing
</p>
</body></html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"  HTML report: {out_path}")


def main():
    from typing import List

    print("=" * 60)
    print("Phase 1.4: Change-Point Detection (MISIS Temporal)")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("\n[1/6] Loading MISIS temporal data...")
    df = pd.read_csv("data/raw/misis/parsed/all_courses.csv")
    print(f"  Total rows: {len(df)}")
    print(f"  Years: {sorted(df['year'].unique())}")

    # Identify programs with 3+ years
    fgos_years = (
        df.groupby("fgos_code")["year"]
        .apply(lambda x: sorted(x.unique()))
        .reset_index()
    )
    fgos_years["n_years"] = fgos_years["year"].apply(len)
    eligible = fgos_years[fgos_years["n_years"] >= 3]
    print(f"  FGOS codes with 3+ years: {len(eligible)}")

    # 2. Build signals
    print("\n[2/6] Building temporal signals per program...")
    evolutions: List[ProgramEvolution] = []
    for _, row in eligible.iterrows():
        fgos = row["fgos_code"]
        evo = build_program_signals(df, fgos)
        evolutions.append(evo)
        n_sig = sum(1 for v in evo.signals.values() if v)
        print(f"  {fgos}: {evo.n_years} years, {n_sig} signals")

    # 3. Detect changepoints
    print("\n[3/6] Running change-point detection (PELT, penalty=3.0)...")
    for evo in evolutions:
        detect_program_changepoints(evo, method="pelt", penalty=3.0)
        if evo.has_changepoints:
            cp_summary = ", ".join(
                f"{cp.signal_name}@{cp.year_before}→{cp.year_after}"
                for cp in evo.changepoints
            )
            print(f"  {evo.fgos_code}: {len(evo.changepoints)} CPs — {cp_summary}")

    progs_with_cp = [e for e in evolutions if e.has_changepoints]
    print(f"\n  Programs with changepoints: {len(progs_with_cp)}/{len(evolutions)}")

    # 4. Permutation testing on programs with changepoints
    print("\n[4/6] Permutation testing (200 permutations)...")
    perm_results_map = {}
    for evo in progs_with_cp:
        for sig_name, values in evo.signals.items():
            cps_for_signal = [cp for cp in evo.changepoints if cp.signal_name == sig_name]
            if not cps_for_signal:
                continue
            perm_result = permutation_test_changepoint(
                signal=values,
                years=evo.years,
                method="pelt",
                penalty=3.0,
                n_permutations=200,
            )
            key = f"{evo.fgos_code}_{sig_name}"
            perm_results_map[key] = perm_result
            for pcp in perm_result["changepoints"]:
                sig_str = "SIGNIFICANT" if pcp["significant"] else "not significant"
                print(
                    f"  {evo.fgos_code}/{sig_name}: "
                    f"idx={pcp['index']}, p={pcp['p_value']:.3f} ({sig_str})"
                )

    # 5. Aggregate results
    print("\n[5/6] Aggregating results...")
    agg = aggregate_changepoints(evolutions)
    print(f"  Consensus years: {agg.get('consensus_years', {})}")

    # 6. Generate figures
    print("\n[6/6] Generating figures and report...")
    figures_b64 = {}

    # Per-program signal plots
    for evo in progs_with_cp:
        out_path = OUT_DIR / f"program_{evo.fgos_code}.png"
        b64 = plot_program_signals(evo, out_path)
        figures_b64[f"program_{evo.fgos_code}"] = b64

    # Also plot top programs without changepoints for comparison (up to 3)
    no_cp = [e for e in evolutions if not e.has_changepoints and e.n_years >= 4][:3]
    for evo in no_cp:
        out_path = OUT_DIR / f"program_{evo.fgos_code}_stable.png"
        plot_program_signals(evo, out_path)

    # Consensus heatmap
    b64 = plot_consensus_heatmap(evolutions, OUT_DIR / "consensus_heatmap.png")
    figures_b64["consensus_heatmap"] = b64

    # Aggregate timeline
    b64 = plot_aggregate_timeline(agg, OUT_DIR / "aggregate_timeline.png")
    figures_b64["aggregate_timeline"] = b64

    # Signal distributions
    b64 = plot_signal_distributions(evolutions, OUT_DIR / "signal_distributions.png")
    figures_b64["signal_distributions"] = b64

    # Permutation test plots
    for key, perm_result in perm_results_map.items():
        fgos_code, sig_name = key.rsplit("_", 1)
        out_path = OUT_DIR / f"perm_{fgos_code}_{sig_name}.png"
        b64 = plot_permutation_results(perm_result, fgos_code, sig_name, out_path)
        figures_b64[f"perm_{fgos_code}_{sig_name}"] = b64

    # HTML report
    generate_html_report(
        evolutions, agg, perm_results_map, figures_b64,
        OUT_DIR / "CHANGEPOINT_REPORT.html",
    )

    # JSON summary
    summary = {
        "generated": pd.Timestamp.now().isoformat(),
        "n_programs_analyzed": len(evolutions),
        "n_programs_with_changepoints": len(progs_with_cp),
        "aggregate": agg,
        "programs": [evo.to_dict() for evo in evolutions],
        "permutation_tests": {
            k: {
                "n_permutations": v["n_permutations"],
                "changepoints": v["changepoints"],
            }
            for k, v in perm_results_map.items()
        },
    }
    summary_path = OUT_DIR / "changepoint_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"  Summary: {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Programs analyzed:      {len(evolutions)}")
    print(f"With changepoints:      {len(progs_with_cp)}")
    print(f"Total changepoints:     {agg['total_changepoints']}")
    print(f"Consensus years:        {agg.get('consensus_years', {})}")
    print(f"\nPer-year breakdown:")
    for y, info in sorted(agg.get("per_year", {}).items()):
        mark = " ★ CONSENSUS" if y in agg.get("consensus_years", {}) else ""
        print(f"  {y}: {info['count']} changepoints across {len(info['programs'])} programs{mark}")
    print(f"\nFigures: {OUT_DIR}/")
    print(f"Report:  {OUT_DIR / 'CHANGEPOINT_REPORT.html'}")


if __name__ == "__main__":
    main()
