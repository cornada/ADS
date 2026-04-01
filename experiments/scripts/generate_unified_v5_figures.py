#!/usr/bin/env python3
"""Generate paper figures from unified_v5 experiment results.

Produces:
  F1: Pareto scatter (market vs mission), colored by university
  F2: 4-obj vs 5-obj Pareto front size comparison bar chart
  F3: Radar chart of objective means per university (Pareto courses only)
  F4: University distribution in Pareto front (4-obj vs 5-obj)
  F5: Competency score distribution (Pareto vs non-Pareto)
  F6: Cross-national market alignment heatmap
"""
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.style.use('seaborn-v0_8-whitegrid')

DPI = 400
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

LEGEND_PROP = {'weight': 'bold', 'size': 11}

COLORS = {
    'MIT': '#E41A1C',
    'UC Berkeley': '#377EB8',
    'Stanford': '#4DAF4A',
    'KTH': '#984EA3',
    'Cornell': '#FF7F00',
    'UIUC': '#A65628',
    'Edinburgh': '#F781BF',
    'MISIS': '#1B9E77',
}

BASE = Path("/Users/aleksandrvolkov/Desktop/ADS")
CORE_DIR = BASE / "experiments/reports/unified_v5_core_multilingual/paper_artifacts/paper_tables"
COMP_DIR = BASE / "experiments/reports/unified_v5_competency_multilingual/paper_artifacts/paper_tables"
OUTPUT_DIR = BASE / "experiments/reports/unified_v5_figures"


def load_csv(path):
    """Load CSV as list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def get_university(option_id):
    """Extract university from option_id like 'course:Stanford:abc123'."""
    parts = option_id.split(":")
    if len(parts) >= 2:
        uni = parts[1]
        if uni == "UC":
            return "UC Berkeley"
        return uni
    return "Unknown"


def generate_f1_pareto_scatter(output_dir):
    """F1: Market vs Mission scatter, Pareto in front, colored by university."""
    pareto = load_csv(CORE_DIR / "pareto_table.csv")
    all_data = load_csv(CORE_DIR / "all_results.csv")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background: all non-Pareto courses
    pareto_ids = {r['option_id'] for r in pareto}
    non_pareto = [r for r in all_data if r['option_id'] not in pareto_ids]

    ax.scatter(
        [float(r['market']) for r in non_pareto],
        [float(r['mission']) for r in non_pareto],
        c='#cccccc', alpha=0.15, s=6, zorder=1, label=f'Non-Pareto (n={len(non_pareto)})'
    )

    # Pareto courses by university
    uni_groups = defaultdict(list)
    for r in pareto:
        uni_groups[get_university(r['option_id'])].append(r)

    for uni in sorted(uni_groups.keys()):
        courses = uni_groups[uni]
        color = COLORS.get(uni, '#999999')
        ax.scatter(
            [float(r['market']) for r in courses],
            [float(r['mission']) for r in courses],
            c=color, s=60, zorder=3, label=f'{uni} ({len(courses)})',
            edgecolors='black', linewidths=0.5, alpha=0.85
        )

    ax.set_xlabel('Market Alignment Score')
    ax.set_ylabel('Mission Alignment Score')
    ax.set_title('Pareto Front: Market vs Mission (unified_v5)')
    ax.legend(loc='lower left', prop=LEGEND_PROP, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / "F1_pareto_scatter_market_mission.png", dpi=DPI)
    plt.close(fig)
    print(f"  F1 saved: {output_dir / 'F1_pareto_scatter_market_mission.png'}")


def generate_f2_4obj_vs_5obj(output_dir):
    """F2: Bar chart comparing 4-objective vs 5-objective Pareto front."""
    with open(CORE_DIR / "summary.json") as f:
        core = json.load(f)
    with open(COMP_DIR / "summary.json") as f:
        comp = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Pareto front size
    ax = axes[0]
    labels = ['4-Objective', '5-Objective\n(+Competency)']
    values = [core['num_pareto'], comp['num_pareto']]
    colors = ['#377EB8', '#E41A1C']
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax.set_ylabel('Pareto-Optimal Courses')
    ax.set_title('Pareto Front Size')

    # Right: University distribution comparison
    ax = axes[1]
    core_pareto = load_csv(CORE_DIR / "pareto_table.csv")
    comp_pareto = load_csv(COMP_DIR / "pareto_table.csv")

    core_unis = defaultdict(int)
    comp_unis = defaultdict(int)
    for r in core_pareto:
        core_unis[get_university(r['option_id'])] += 1
    for r in comp_pareto:
        comp_unis[get_university(r['option_id'])] += 1

    all_unis = sorted(set(core_unis) | set(comp_unis), key=lambda u: -comp_unis.get(u, 0))
    x = np.arange(len(all_unis))
    width = 0.35

    ax.bar(x - width/2, [core_unis.get(u, 0) for u in all_unis],
           width, label='4-Objective', color='#377EB8', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, [comp_unis.get(u, 0) for u in all_unis],
           width, label='5-Objective', color='#E41A1C', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_unis, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Pareto Front by University')
    ax.legend(prop=LEGEND_PROP)

    fig.tight_layout()
    fig.savefig(output_dir / "F2_4obj_vs_5obj_comparison.png", dpi=DPI)
    plt.close(fig)
    print(f"  F2 saved: {output_dir / 'F2_4obj_vs_5obj_comparison.png'}")


def generate_f3_radar_by_university(output_dir):
    """F3: Radar chart of mean objective scores for Pareto courses, by university."""
    pareto = load_csv(COMP_DIR / "pareto_table.csv")
    objectives = ['market', 'mission', 'university', 'learner', 'competency']

    uni_scores = defaultdict(lambda: defaultdict(list))
    for r in pareto:
        uni = get_university(r['option_id'])
        for obj in objectives:
            uni_scores[uni][obj].append(float(r[obj]))

    # Compute means
    uni_means = {}
    for uni, scores in uni_scores.items():
        uni_means[uni] = [np.mean(scores[obj]) for obj in objectives]

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(objectives), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for uni in sorted(uni_means.keys(), key=lambda u: -len(uni_scores[u]['market'])):
        vals = uni_means[uni] + uni_means[uni][:1]
        color = COLORS.get(uni, '#999999')
        ax.plot(angles, vals, 'o-', linewidth=2, color=color, label=uni, markersize=5)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), objectives)
    ax.set_ylim(0, 0.85)
    ax.set_title('Objective Specialization by University\n(Pareto courses, 5-objective)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), prop=LEGEND_PROP)
    fig.tight_layout()
    fig.savefig(output_dir / "F3_radar_university_specialization.png", dpi=DPI)
    plt.close(fig)
    print(f"  F3 saved: {output_dir / 'F3_radar_university_specialization.png'}")


def generate_f4_competency_distribution(output_dir):
    """F4: Competency score distribution: Pareto vs non-Pareto."""
    pareto = load_csv(COMP_DIR / "pareto_table.csv")
    all_data = load_csv(COMP_DIR / "all_results.csv")
    pareto_ids = {r['option_id'] for r in pareto}

    pareto_comp = [float(r['competency']) for r in pareto]
    non_pareto_comp = [float(r['competency']) for r in all_data if r['option_id'] not in pareto_ids]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(non_pareto_comp, bins=50, alpha=0.6, color='#cccccc', label=f'Non-Pareto (n={len(non_pareto_comp)})', edgecolor='gray')
    ax.hist(pareto_comp, bins=20, alpha=0.8, color='#E41A1C', label=f'Pareto (n={len(pareto_comp)})', edgecolor='black')
    ax.axvline(np.mean(non_pareto_comp), color='gray', linestyle='--', linewidth=1.5, label=f'Non-Pareto mean={np.mean(non_pareto_comp):.3f}')
    ax.axvline(np.mean(pareto_comp), color='#E41A1C', linestyle='--', linewidth=1.5, label=f'Pareto mean={np.mean(pareto_comp):.3f}')
    ax.set_xlabel('Competency Alignment Score')
    ax.set_ylabel('Count')
    ax.set_title('Competency Score Distribution: Pareto vs Non-Pareto')
    ax.legend(prop=LEGEND_PROP)
    fig.tight_layout()
    fig.savefig(output_dir / "F4_competency_distribution.png", dpi=DPI)
    plt.close(fig)
    print(f"  F4 saved: {output_dir / 'F4_competency_distribution.png'}")


def generate_f5_cross_national_heatmap(output_dir):
    """F5: Cross-national heatmap of mean objective scores by university."""
    all_data = load_csv(COMP_DIR / "all_results.csv")
    objectives = ['market', 'mission', 'university', 'learner', 'competency']

    uni_scores = defaultdict(lambda: defaultdict(list))
    for r in all_data:
        uni = get_university(r['option_id'])
        for obj in objectives:
            uni_scores[uni][obj].append(float(r[obj]))

    # Filter to universities with >10 courses
    unis = sorted([u for u in uni_scores if len(uni_scores[u]['market']) > 10],
                  key=lambda u: -np.mean(uni_scores[u]['market']))

    matrix = np.array([[np.mean(uni_scores[u][obj]) for obj in objectives] for u in unis])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=-0.1, vmax=0.7)

    ax.set_xticks(range(len(objectives)))
    ax.set_xticklabels([o.capitalize() for o in objectives], rotation=45, ha='right')
    ax.set_yticks(range(len(unis)))
    ax.set_yticklabels(unis)

    # Add text annotations
    for i in range(len(unis)):
        for j in range(len(objectives)):
            val = matrix[i, j]
            color = 'white' if val > 0.5 or val < 0.1 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=10, fontweight='bold')

    ax.set_title('Cross-National Objective Alignment\n(Mean score per university, all courses)')
    fig.colorbar(im, ax=ax, label='Cosine Similarity', shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "F5_cross_national_heatmap.png", dpi=DPI)
    plt.close(fig)
    print(f"  F5 saved: {output_dir / 'F5_cross_national_heatmap.png'}")


def generate_f6_pareto_front_expansion(output_dir):
    """F6: Scatter showing which courses are NEW in 5-obj Pareto vs 4-obj."""
    core_pareto = load_csv(CORE_DIR / "pareto_table.csv")
    comp_pareto = load_csv(COMP_DIR / "pareto_table.csv")

    core_ids = {r['option_id'] for r in core_pareto}
    comp_ids = {r['option_id'] for r in comp_pareto}

    shared = core_ids & comp_ids
    new_in_5obj = comp_ids - core_ids
    dropped = core_ids - comp_ids

    # Build lookup for comp_pareto by id
    comp_lookup = {r['option_id']: r for r in comp_pareto}
    core_lookup = {r['option_id']: r for r in core_pareto}

    fig, ax = plt.subplots(figsize=(10, 8))

    # Shared courses
    shared_data = [comp_lookup[oid] for oid in shared if oid in comp_lookup]
    ax.scatter(
        [float(r['market']) for r in shared_data],
        [float(r['competency']) for r in shared_data],
        c='#377EB8', s=50, alpha=0.7, label=f'Both fronts ({len(shared)})',
        edgecolors='black', linewidths=0.5, zorder=3
    )

    # New in 5-obj
    new_data = [comp_lookup[oid] for oid in new_in_5obj if oid in comp_lookup]
    ax.scatter(
        [float(r['market']) for r in new_data],
        [float(r['competency']) for r in new_data],
        c='#E41A1C', s=80, marker='*', alpha=0.8, label=f'New in 5-obj ({len(new_in_5obj)})',
        edgecolors='black', linewidths=0.5, zorder=4
    )

    ax.set_xlabel('Market Alignment Score')
    ax.set_ylabel('Competency Alignment Score')
    ax.set_title('Pareto Front Expansion: 4-obj → 5-obj')
    ax.legend(prop=LEGEND_PROP, loc='lower right')
    fig.tight_layout()
    fig.savefig(output_dir / "F6_pareto_expansion_market_competency.png", dpi=DPI)
    plt.close(fig)
    print(f"  F6 saved: {output_dir / 'F6_pareto_expansion_market_competency.png'}")


def print_summary():
    """Print key findings summary."""
    with open(CORE_DIR / "summary.json") as f:
        core = json.load(f)
    with open(COMP_DIR / "summary.json") as f:
        comp = json.load(f)

    print("\n" + "="*60)
    print("UNIFIED V5 EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Encoder: {core['encoder']}")
    print(f"Total options evaluated: {core['num_options']:,}")
    print(f"Feasible (tau < 0.25): {core['feasible_count']:,}")
    print()
    print("4-Objective (market, mission, university, learner):")
    print(f"  Pareto front: {core['num_pareto']} ({core['pareto_ratio']*100:.2f}%)")
    for obj, data in core['best_per_objective'].items():
        print(f"  Best {obj}: {data['option_id']} ({data['score']:.4f})")
    print()
    print("5-Objective (+competency):")
    print(f"  Pareto front: {comp['num_pareto']} ({comp['pareto_ratio']*100:.2f}%)")
    for obj, data in comp['best_per_objective'].items():
        print(f"  Best {obj}: {data['option_id']} ({data['score']:.4f})")
    print()
    print(f"Pareto expansion: +{comp['num_pareto'] - core['num_pareto']} courses (+{(comp['num_pareto']/core['num_pareto']-1)*100:.0f}%)")
    print("="*60)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating unified_v5 paper figures...")
    generate_f1_pareto_scatter(OUTPUT_DIR)
    generate_f2_4obj_vs_5obj(OUTPUT_DIR)
    generate_f3_radar_by_university(OUTPUT_DIR)
    generate_f4_competency_distribution(OUTPUT_DIR)
    generate_f5_cross_national_heatmap(OUTPUT_DIR)
    generate_f6_pareto_front_expansion(OUTPUT_DIR)

    print_summary()
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
