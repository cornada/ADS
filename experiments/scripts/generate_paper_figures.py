"""Generate paper figures F1-F6 from existing experiment results."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Use non-interactive backend for headless execution
import matplotlib
matplotlib.use('Agg')

def generate_f1_scatter(run_dir, output_dir):
    """F1: Scatter plot of market vs mission with edge cases marked."""
    print("Generating F1: Market-Mission scatter with edge cases...")

    cache_dir = run_dir / "cache_embeddings" / "sbert_sentence-transformers_all-MiniLM-L6-v2"
    embeddings = np.load(cache_dir / "vecs.npy")
    with open(cache_dir / "index.json", encoding='utf-8') as f:
        index = json.load(f)

    with open(run_dir / "store" / "artifacts.jsonl", encoding='utf-8') as f:
        artifacts = [json.loads(line) for line in f]

    from hashlib import sha256

    text_to_idx = {}
    for a in artifacts:
        h = sha256(a['text'].encode()).hexdigest()
        if h in index:
            text_to_idx[a['artifact_id']] = index[h]

    courses = [a for a in artifacts if a['type'] == 'COURSE']
    jobs = [a for a in artifacts if a['type'] == 'JOB_ROLE']
    missions = [a for a in artifacts if a['type'] == 'MISSION']

    job_embs = np.array([embeddings[text_to_idx[j['artifact_id']]]
                         for j in jobs if j['artifact_id'] in text_to_idx])
    mission_embs = np.array([embeddings[text_to_idx[m['artifact_id']]]
                             for m in missions if m['artifact_id'] in text_to_idx])

    market_c = np.mean(job_embs, axis=0)
    market_c = market_c / np.linalg.norm(market_c)
    mission_c = np.mean(mission_embs, axis=0)
    mission_c = mission_c / np.linalg.norm(mission_c)

    course_scores = []
    for c in courses:
        if c['artifact_id'] in text_to_idx:
            emb = embeddings[text_to_idx[c['artifact_id']]]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            course_scores.append({
                'id': c['artifact_id'],
                'market': float(np.dot(emb_norm, market_c)),
                'mission': float(np.dot(emb_norm, mission_c)),
                'university': 'UCB' if ':UC Berkeley:' in c['artifact_id'] else 'MIT'
            })

    # Compute Pareto
    pareto_ids = set()
    for i, c_i in enumerate(course_scores):
        is_pareto = True
        for j, c_j in enumerate(course_scores):
            if i != j:
                if c_j['market'] >= c_i['market'] and c_j['mission'] >= c_i['mission']:
                    if c_j['market'] > c_i['market'] or c_j['mission'] > c_i['mission']:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_ids.add(c_i['id'])

    pareto_courses = [c for c in course_scores if c['id'] in pareto_ids]

    # Find extremes
    max_market = max(pareto_courses, key=lambda x: x['market'])
    max_mission = max(pareto_courses, key=lambda x: x['mission'])
    balanced = min(pareto_courses, key=lambda x: abs(x['market'] - x['mission']))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # All courses (faded)
    all_markets = [c['market'] for c in course_scores]
    all_missions = [c['mission'] for c in course_scores]
    ax.scatter(all_markets, all_missions, alpha=0.1, c='gray', s=10, label='All courses')

    # Pareto frontier
    pareto_markets = [c['market'] for c in pareto_courses]
    pareto_missions = [c['mission'] for c in pareto_courses]

    ucb_pareto = [(c['market'], c['mission']) for c in pareto_courses if c['university'] == 'UCB']
    mit_pareto = [(c['market'], c['mission']) for c in pareto_courses if c['university'] == 'MIT']

    if ucb_pareto:
        ax.scatter([p[0] for p in ucb_pareto], [p[1] for p in ucb_pareto],
                   c='blue', s=100, marker='o', label=f'UCB Pareto ({len(ucb_pareto)})', edgecolors='black')
    if mit_pareto:
        ax.scatter([p[0] for p in mit_pareto], [p[1] for p in mit_pareto],
                   c='red', s=100, marker='s', label=f'MIT Pareto ({len(mit_pareto)})', edgecolors='black')

    # Mark extremes
    ax.scatter([max_market['market']], [max_market['mission']],
               c='green', s=200, marker='*', label='Max Market', zorder=5)
    ax.scatter([max_mission['market']], [max_mission['mission']],
               c='purple', s=200, marker='*', label='Max Mission', zorder=5)
    ax.scatter([balanced['market']], [balanced['mission']],
               c='orange', s=200, marker='*', label='Most Balanced', zorder=5)

    ax.set_xlabel('Market Alignment Score', fontsize=12)
    ax.set_ylabel('Mission Alignment Score', fontsize=12)
    ax.set_title('ADS-2D: Market vs Mission Pareto Frontier', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'F1_scatter_market_mission_edge_cases.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'F1_scatter_market_mission_edge_cases.png'}")

def generate_f2_tau_sweep(output_dir):
    """F2: Tau sweep curve."""
    print("Generating F2: Tau sweep curve...")

    tau_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, float('inf')]
    pareto_counts = [19, 51, 73, 92, 103, 107, 109, 109]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot line
    x_vals = [t if t != float('inf') else 0.6 for t in tau_values]
    ax.plot(x_vals, pareto_counts, 'b-o', linewidth=2, markersize=8)

    # Mark default
    ax.axvline(x=0.25, color='green', linestyle='--', alpha=0.7, label='Default tau=0.25')
    ax.scatter([0.25], [103], c='green', s=150, zorder=5, marker='D')

    ax.set_xlabel('Autonomy Constraint (tau)', fontsize=12)
    ax.set_ylabel('Pareto Set Size', fontsize=12)
    ax.set_title('Effect of Governance Constraint on Pareto Options', fontsize=14)

    # Custom x-ticks
    ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.6])
    ax.set_xticklabels(['0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.50', 'inf'])

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'F2_tau_sweep_curve.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'F2_tau_sweep_curve.png'}")

def generate_f3_scalarization_bar(output_dir):
    """F3: Scalarization vs Pareto bar chart."""
    print("Generating F3: Scalarization vs Pareto comparison...")

    methods = ['ADS-Pareto', 'Weighted-Sum\n(50x grid)', 'Random\nScalarization', 'Lexicographic\n(best)', 'Single-Obj\nTop-10']
    solutions = [103, 4, 4, 10, 10]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green', 'red', 'red', 'orange', 'orange']
    bars = ax.bar(methods, solutions, color=colors, edgecolor='black')

    # Add value labels
    for bar, val in zip(bars, solutions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of Unique Solutions', fontsize=12)
    ax.set_title('Multi-Objective Methods Comparison', fontsize=14)
    ax.set_ylim(0, 120)
    ax.axhline(y=103, color='green', linestyle='--', alpha=0.5, label='ADS-Pareto baseline')

    plt.tight_layout()
    plt.savefig(output_dir / 'F3_scalarization_vs_pareto_bar.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'F3_scalarization_vs_pareto_bar.png'}")

def generate_f4_stability_jaccard(output_dir):
    """F4: Stability Jaccard vs epsilon."""
    print("Generating F4: Stability Jaccard vs epsilon...")

    epsilons = [0.01, 0.02, 0.05, 0.10]
    mean_jaccards = [0.071, 0.068, 0.043, 0.014]
    std_jaccards = [0.008, 0.012, 0.014, 0.010]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(epsilons, mean_jaccards, yerr=std_jaccards,
                fmt='o-', linewidth=2, markersize=10, capsize=5, color='blue')

    ax.set_xlabel('Perturbation Epsilon', fontsize=12)
    ax.set_ylabel('Jaccard Similarity with Base Pareto', fontsize=12)
    ax.set_title('Pareto Set Stability Under Score Perturbation', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Higher epsilon = more noise\n-> lower stability',
                xy=(0.05, 0.043), xytext=(0.07, 0.055),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / 'F4_stability_jaccard_vs_eps.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'F4_stability_jaccard_vs_eps.png'}")

def generate_f5_cross_university(output_dir):
    """F5: Cross-university overlap Venn-style diagram."""
    print("Generating F5: Cross-university overlap...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    labels = ['Full\n(Pooled)', 'UC Berkeley\nOnly', 'MIT\nOnly']
    pareto_counts = [11, 7, 15]
    colors = ['green', 'blue', 'red']

    x_pos = [0, 1, 2]
    bars = ax.bar(x_pos, pareto_counts, color=colors, edgecolor='black', alpha=0.7)

    for bar, val in zip(bars, pareto_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Pareto Set Size', fontsize=12)
    ax.set_title('Cross-University Validation (ADS-2D)', fontsize=14)

    # Add Jaccard annotation
    ax.text(0.5, 0.95, 'Jaccard(UCB, MIT) = 0.000\nNo overlap between university-specific Pareto sets',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11)

    ax.set_ylim(0, 20)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'F5_cross_university_overlap.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'F5_cross_university_overlap.png'}")

def generate_f6_scaling(output_dir):
    """F6: Scaling curve."""
    print("Generating F6: Scaling curve...")

    n_values = [500, 1000, 2000, 5000, 10000, 18000]
    times_ms = [36.6, 79.1, 207.1, 582.0, 1159.5, 2921.4]
    times_s = [t/1000 for t in times_ms]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(n_values, times_s, 'bo-', linewidth=2, markersize=10, label='Measured')

    # Fit O(n^1.2) line
    fit_n = np.linspace(500, 18000, 100)
    fit_time = (fit_n / 500) ** 1.2 * times_s[0]
    ax.plot(fit_n, fit_time, 'r--', linewidth=2, alpha=0.7, label=r'$O(n^{1.2})$ fit')

    # Extrapolation
    extrap_n = [50000, 100000]
    extrap_t = [(n / 500) ** 1.2 * times_s[0] for n in extrap_n]
    ax.scatter(extrap_n, extrap_t, c='orange', s=100, marker='x', zorder=5, label='Extrapolated')

    ax.set_xlabel('Number of Options (n)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Computational Scaling Analysis', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate(f'50K: ~9.2s\n100K: ~21s',
                xy=(75000, 15), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'F6_scaling_curve.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'F6_scaling_curve.png'}")

def main():
    import argparse
    _REPO_ROOT = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Generate paper figures F1-F6")
    parser.add_argument('--run-dir', type=Path,
                        default=_REPO_ROOT / 'experiments/reports/latest',
                        help='Path to experiment run directory')
    parser.add_argument('--output-dir', type=Path,
                        default=_REPO_ROOT / 'experiments/reports/figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    run_dir = args.run_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating paper figures to {output_dir}...\n")

    generate_f1_scatter(run_dir, output_dir)
    generate_f2_tau_sweep(output_dir)
    generate_f3_scalarization_bar(output_dir)
    generate_f4_stability_jaccard(output_dir)
    generate_f5_cross_university(output_dir)
    generate_f6_scaling(output_dir)

    print(f"\nAll figures generated successfully!")

if __name__ == "__main__":
    main()
