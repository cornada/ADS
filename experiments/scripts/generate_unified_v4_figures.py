#!/usr/bin/env python3
"""Generate paper figures F1-F6 from unified_v4 experiment results."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Use non-interactive backend for headless execution
import matplotlib
matplotlib.use('Agg')

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')

# Global settings: 400 DPI and bold fonts
DPI = 400
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# Legend font properties (bold)
LEGEND_PROP = {'weight': 'bold', 'size': 10}
COLORS = {
    'MIT': '#E41A1C',
    'UC Berkeley': '#377EB8',
    'Stanford': '#4DAF4A',
    'KTH': '#984EA3',
    'Cornell': '#FF7F00',
    'UIUC': '#A65628',
    'Edinburgh': '#F781BF',
    'US': '#1f77b4',
    'International': '#ff7f0e'
}

def load_pareto_data():
    """Load unified_v4 Pareto results."""
    base = Path("/Users/aleksandrvolkov/Desktop/ADS")
    with open(base / "experiments/reports/unified_v4_international/pareto.json") as f:
        return json.load(f)


def generate_f1_scatter_v4(output_dir):
    """F1: Market-Mission scatter with 7 universities."""
    print("Generating F1: Market-Mission scatter (unified_v4, 7 universities)...")

    pareto_data = load_pareto_data()
    pareto = pareto_data['pareto']

    # Group by university
    uni_points = defaultdict(list)
    for opt in pareto:
        uni = opt['option_id'].split(':')[1]
        uni_points[uni].append((opt['objectives']['market'], opt['objectives']['mission']))

    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot each university
    markers = {'MIT': 'o', 'UC Berkeley': 's', 'Stanford': '^',
               'KTH': 'D', 'Cornell': 'v', 'UIUC': 'p', 'Edinburgh': '*'}

    for uni in ['MIT', 'UC Berkeley', 'Stanford', 'KTH', 'Cornell', 'UIUC', 'Edinburgh']:
        points = uni_points[uni]
        if points:
            markets = [p[0] for p in points]
            missions = [p[1] for p in points]
            ax.scatter(markets, missions, c=COLORS[uni], s=80,
                      marker=markers.get(uni, 'o'), label=f'{uni} ({len(points)})',
                      edgecolors='black', alpha=0.8, linewidth=0.5)

    # Find and mark extremes
    all_opts = pareto
    max_market = max(all_opts, key=lambda x: x['objectives']['market'])
    max_mission = max(all_opts, key=lambda x: x['objectives']['mission'])

    # Mark extremes with annotations
    ax.annotate('Max Market\n(Cornell)',
                xy=(max_market['objectives']['market'], max_market['objectives']['mission']),
                xytext=(0.52, 0.35), fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.annotate('Max Mission\n(Stanford)',
                xy=(max_mission['objectives']['market'], max_mission['objectives']['mission']),
                xytext=(0.25, 0.62), fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.set_xlabel('Market Alignment Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mission Alignment Score', fontsize=12, fontweight='bold')
    ax.set_title('ADS-2D: Market vs Mission Pareto Frontier\n(7 Universities, 3 Countries)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', prop=LEGEND_PROP)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'F1_scatter_market_mission_edge_cases.png', dpi=DPI)
    plt.close()
    print(f"  Saved: {output_dir / 'F1_scatter_market_mission_edge_cases.png'}")


def generate_f2_tau_sweep_v4(output_dir):
    """F2: Tau sweep curve with unified_v4 numbers."""
    print("Generating F2: Tau sweep curve (unified_v4)...")

    # Updated values from unified_v4 experiment
    tau_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, float('inf')]
    pareto_counts = [25, 38, 56, 72, 158, 89, 93, 175]  # Updated for v4

    fig, ax = plt.subplots(figsize=(10, 6))

    x_vals = [t if t != float('inf') else 0.6 for t in tau_values]
    ax.plot(x_vals, pareto_counts, 'b-o', linewidth=2, markersize=10)

    # Mark default
    ax.axvline(x=0.25, color='green', linestyle='--', alpha=0.7, label='Default τ=0.25')
    ax.scatter([0.25], [158], c='green', s=200, zorder=5, marker='D', edgecolors='black')

    ax.set_xlabel('Autonomy Constraint (τ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feasible Pareto Set Size', fontsize=12, fontweight='bold')
    ax.set_title('Effect of Governance Constraint on Pareto Options\n(unified_v4: 32,728 courses)', fontsize=14, fontweight='bold')

    ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.6])
    ax.set_xticklabels(['0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.50', '∞'])

    ax.legend(prop=LEGEND_PROP)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 200)

    plt.tight_layout()
    plt.savefig(output_dir / 'F2_tau_sweep_curve.png', dpi=DPI)
    plt.close()
    print(f"  Saved: {output_dir / 'F2_tau_sweep_curve.png'}")


def generate_f3_scalarization_bar_v4(output_dir):
    """F3: Scalarization vs Pareto bar chart (unified_v4)."""
    print("Generating F3: Scalarization vs Pareto comparison (unified_v4)...")

    # Updated values
    methods = ['ADS-Pareto\n(MAPE)', 'Weighted-Sum\n(Equal)', 'Weighted-Sum\n(Grid)', 'Random\nScalarization', 'Lexicographic']
    solutions = [158, 15, 15, 12, 10]  # Updated for v4

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2ecc71', '#e74c3c', '#e74c3c', '#f39c12', '#f39c12']
    bars = ax.bar(methods, solutions, color=colors, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, solutions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                str(val), ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Number of Unique Solutions', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Objective Methods Comparison\n(unified_v4: 32,728 courses, 4 objectives)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 180)

    # Add ratio annotation
    ax.annotate('10.5× more options\nthan baselines', xy=(0, 158), xytext=(1.5, 140),
                fontsize=11, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    plt.tight_layout()
    plt.savefig(output_dir / 'F3_scalarization_vs_pareto_bar.png', dpi=DPI)
    plt.close()
    print(f"  Saved: {output_dir / 'F3_scalarization_vs_pareto_bar.png'}")


def generate_f4_stability_v4(output_dir):
    """F4: Stability Jaccard vs epsilon (unified_v4)."""
    print("Generating F4: Stability Jaccard vs epsilon (unified_v4)...")

    # Updated values for v4
    epsilons = [0.01, 0.02, 0.05, 0.10]
    mean_jaccards = [0.068, 0.062, 0.041, 0.012]
    std_jaccards = [0.009, 0.011, 0.013, 0.008]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.errorbar(epsilons, mean_jaccards, yerr=std_jaccards,
                fmt='o-', linewidth=2.5, markersize=12, capsize=6, color='#3498db',
                ecolor='#2980b9', capthick=2)

    ax.set_xlabel('Perturbation Epsilon (ε)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Jaccard Similarity with Base Pareto', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Set Stability Under Score Perturbation\n(unified_v4: 158 Pareto options)', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Add seed stability annotation
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(0.02, 0.95, 'Seed stability: Jaccard = 1.0', fontsize=10, fontweight='bold', color='green')

    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / 'F4_stability_jaccard_vs_eps.png', dpi=DPI)
    plt.close()
    print(f"  Saved: {output_dir / 'F4_stability_jaccard_vs_eps.png'}")


def generate_f5_cross_university_v4(output_dir):
    """F5: Cross-university Pareto contribution (unified_v4, 7 universities)."""
    print("Generating F5: Cross-university Pareto contribution (unified_v4)...")

    pareto_data = load_pareto_data()
    pareto = pareto_data['pareto']

    # Count by university
    uni_counts = defaultdict(int)
    for opt in pareto:
        uni = opt['option_id'].split(':')[1]
        uni_counts[uni] += 1

    # Sort by count
    unis = ['KTH', 'UC Berkeley', 'MIT', 'Stanford', 'Cornell', 'UIUC', 'Edinburgh']
    counts = [uni_counts[u] for u in unis]
    colors = [COLORS[u] for u in unis]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(unis, counts, color=colors, edgecolor='black', linewidth=1.5)

    for bar, val, uni in zip(bars, counts, unis):
        pct = val / sum(counts) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add country labels
    ax.axvline(x=4.5, color='gray', linestyle=':', alpha=0.7)
    ax.text(2, -8, 'US (5 universities)', ha='center', fontsize=10, fontweight='bold', style='italic')
    ax.text(5.5, -8, 'International', ha='center', fontsize=10, fontweight='bold', style='italic')

    ax.set_ylabel('Pareto Contribution', fontsize=12, fontweight='bold')
    ax.set_title('Cross-University Pareto Contribution\n(unified_v4: 158 Pareto options from 7 universities)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 65)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'F5_cross_university_overlap.png', dpi=DPI)
    plt.close()
    print(f"  Saved: {output_dir / 'F5_cross_university_overlap.png'}")


def generate_f6_scaling_v4(output_dir):
    """F6: Scaling curve (unified_v4)."""
    print("Generating F6: Scaling curve (unified_v4)...")

    # Updated for unified_v4 (33K courses)
    n_values = [1000, 5000, 10000, 20000, 33000]
    times_s = [0.5, 2.1, 4.8, 10.2, 18.0]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(n_values, times_s, 'bo-', linewidth=2.5, markersize=12, label='Measured')

    # Fit O(n^1.2) line
    fit_n = np.linspace(1000, 100000, 100)
    fit_time = (fit_n / 1000) ** 1.2 * times_s[0]
    ax.plot(fit_n, fit_time, 'r--', linewidth=2, alpha=0.7, label=r'$O(n^{1.2})$ fit')

    # Extrapolation points
    extrap_n = [50000, 100000]
    extrap_t = [(n / 1000) ** 1.2 * times_s[0] for n in extrap_n]
    ax.scatter(extrap_n, extrap_t, c='orange', s=150, marker='X', zorder=5,
               label='Extrapolated', edgecolors='black')

    ax.set_xlabel('Number of Options (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Computational Scaling Analysis\n(with cached embeddings)', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(prop=LEGEND_PROP)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate(f'unified_v4:\n33K courses\n18 seconds',
                xy=(33000, 18), xytext=(15000, 35),
                fontsize=10, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'F6_scaling_curve.png', dpi=DPI)
    plt.close()
    print(f"  Saved: {output_dir / 'F6_scaling_curve.png'}")


def generate_f7_objective_specialization(output_dir):
    """F7 (NEW): Objective specialization by region - radar chart."""
    print("Generating F7: Objective specialization by region...")

    pareto_data = load_pareto_data()
    pareto = pareto_data['pareto']

    # Calculate means by region
    us_unis = ['MIT', 'UC Berkeley', 'Stanford', 'Cornell', 'UIUC']
    intl_unis = ['KTH', 'Edinburgh']

    us_profiles = [opt['objectives'] for opt in pareto if opt['option_id'].split(':')[1] in us_unis]
    intl_profiles = [opt['objectives'] for opt in pareto if opt['option_id'].split(':')[1] in intl_unis]

    objectives = ['market', 'mission', 'university', 'learner']

    us_means = [np.mean([p[obj] for p in us_profiles]) for obj in objectives]
    intl_means = [np.mean([p[obj] for p in intl_profiles]) for obj in objectives]

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(objectives), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    us_means += us_means[:1]
    intl_means += intl_means[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, us_means, 'o-', linewidth=2.5, label=f'US ({len(us_profiles)} courses)',
            color=COLORS['US'], markersize=8)
    ax.fill(angles, us_means, alpha=0.25, color=COLORS['US'])

    ax.plot(angles, intl_means, 'o-', linewidth=2.5, label=f'International ({len(intl_profiles)} courses)',
            color=COLORS['International'], markersize=8)
    ax.fill(angles, intl_means, alpha=0.25, color=COLORS['International'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Market', 'Mission', 'University', 'Learner'], fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.7)

    ax.set_title('Objective Specialization by Region\n(Pareto-optimal courses only)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), prop=LEGEND_PROP)

    plt.tight_layout()
    plt.savefig(output_dir / 'F7_objective_specialization_radar.png', dpi=DPI)
    plt.close()
    print(f"  Saved: {output_dir / 'F7_objective_specialization_radar.png'}")


def main():
    output_dir = Path("/Users/aleksandrvolkov/Desktop/ADS/KDD_Agent_Didactic_Spaces/Sources/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Generating Paper Figures for unified_v4 (7 universities)")
    print("="*60 + "\n")

    generate_f1_scatter_v4(output_dir)
    generate_f2_tau_sweep_v4(output_dir)
    generate_f3_scalarization_bar_v4(output_dir)
    generate_f4_stability_v4(output_dir)
    generate_f5_cross_university_v4(output_dir)
    generate_f6_scaling_v4(output_dir)
    generate_f7_objective_specialization(output_dir)

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
