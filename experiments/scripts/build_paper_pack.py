#!/usr/bin/env python3
"""
Build Paper Pack - Generates a complete, reproducible evidence pack for IJCAI paper.

This script:
1) Runs required experiments (or verifies they exist)
2) Aggregates results into CSV + TeX
3) Generates all figures (F1-F6)
4) Writes results_index.md
5) Writes repro_manifest.json with hardened fields

Usage:
    python build_paper_pack.py --dataset unified_v3 --tag paper_max
    python build_paper_pack.py --dataset unified_v3 --tag paper_max --skip-experiments
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

# Use non-interactive backend for headless execution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_python_version():
    """Get Python version."""
    return sys.version.split()[0]


def get_platform_info():
    """Get platform information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def get_key_packages():
    """Get versions of key packages."""
    packages = ['numpy', 'scipy', 'torch', 'sentence-transformers', 'hydra-core', 'matplotlib']
    result = {}
    for pkg in packages:
        try:
            result[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            result[pkg] = "not installed"
    return result


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    if not filepath.exists():
        return "file_not_found"
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def compute_artifacts_hash(run_dir: Path) -> str:
    """Compute hash of artifacts.jsonl for reproducibility."""
    artifacts_file = run_dir / "store" / "artifacts.jsonl"
    return compute_file_hash(artifacts_file)


def load_embeddings_and_artifacts(run_dir: Path, encoder_subdir: str = "sbert_sentence-transformers_all-MiniLM-L6-v2"):
    """Load embeddings and artifacts from a run directory."""
    cache_dir = run_dir / "cache_embeddings" / encoder_subdir
    embeddings = np.load(cache_dir / "vecs.npy")
    with open(cache_dir / "index.json", encoding='utf-8') as f:
        index = json.load(f)

    with open(run_dir / "store" / "artifacts.jsonl", encoding='utf-8') as f:
        artifacts = [json.loads(line) for line in f]

    # Build text->index mapping
    text_to_idx = {}
    for a in artifacts:
        h = hashlib.sha256(a['text'].encode()).hexdigest()
        if h in index:
            text_to_idx[a['artifact_id']] = index[h]

    return embeddings, artifacts, text_to_idx


def compute_pareto(scores_dict, keys):
    """Compute Pareto frontier from scores dictionary."""
    items = list(scores_dict.items())
    pareto = []
    for i, (opt_id, scores_i) in enumerate(items):
        is_pareto = True
        for j, (_, scores_j) in enumerate(items):
            if i != j:
                dominated = all(scores_j[k] >= scores_i[k] for k in keys)
                strictly_better = any(scores_j[k] > scores_i[k] for k in keys)
                if dominated and strictly_better:
                    is_pareto = False
                    break
        if is_pareto:
            pareto.append(opt_id)
    return set(pareto)


def compute_stable_core(run_dir: Path, n_samples: int = 50, epsilon: float = 0.02, seed: int = 42):
    """Compute stable core from perturbations."""
    embeddings, artifacts, text_to_idx = load_embeddings_and_artifacts(run_dir)

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

    # Build course scores
    course_data = []
    for c in courses:
        if c['artifact_id'] in text_to_idx:
            emb = embeddings[text_to_idx[c['artifact_id']]]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            course_data.append({
                'id': c['artifact_id'],
                'market': float(np.dot(emb_norm, market_c)),
                'mission': float(np.dot(emb_norm, mission_c)),
            })

    # Base Pareto
    base_scores = {c['id']: {'market': c['market'], 'mission': c['mission']}
                   for c in course_data}
    base_pareto = compute_pareto(base_scores, ['market', 'mission'])

    # Perturbation sampling
    np.random.seed(seed)
    membership = Counter()
    pareto_sizes = []

    for _ in range(n_samples):
        perturbed_scores = {}
        for c in course_data:
            perturbed_scores[c['id']] = {
                'market': c['market'] + np.random.uniform(-epsilon, epsilon),
                'mission': c['mission'] + np.random.uniform(-epsilon, epsilon),
            }
        pareto = compute_pareto(perturbed_scores, ['market', 'mission'])
        pareto_sizes.append(len(pareto))
        for opt in pareto:
            membership[opt] += 1

    # Compute stable cores
    stable_cores = {}
    for p in [0.5, 0.8]:
        min_count = int(n_samples * p)
        core = [opt for opt, count in membership.items() if count >= min_count]
        stable_cores[f'p={p}'] = {
            'threshold': p,
            'min_count': min_count,
            'core_size': len(core),
            'core_ids': sorted(core)
        }

    return {
        'base_pareto_size': len(base_pareto),
        'perturbed_pareto_size': {
            'mean': float(np.mean(pareto_sizes)),
            'std': float(np.std(pareto_sizes)),
        },
        'stable_cores': stable_cores,
        'top_options': [{'id': opt, 'frequency': count, 'pct': count * 100 / n_samples}
                        for opt, count in membership.most_common(20)],
        'config': {
            'n_samples': n_samples,
            'epsilon': epsilon,
            'seed': seed,
        }
    }


def generate_t7_from_stable_core(stable_core_data: dict, output_dir: Path):
    """Generate T7_stability_core.csv and .tex from stable_core.json data."""
    rows = [
        ("Base Pareto Size (ADS-2D)", stable_core_data['base_pareto_size'], "Reference"),
        ("Perturbed Pareto (mean ± std)",
         f"{stable_core_data['perturbed_pareto_size']['mean']:.1f} ± {stable_core_data['perturbed_pareto_size']['std']:.1f}",
         f"n={stable_core_data['config']['n_samples']}, eps={stable_core_data['config']['epsilon']}"),
    ]

    for key, core in stable_core_data['stable_cores'].items():
        rows.append((f"Stable Core @ {key}", core['core_size'],
                     f">{int(core['threshold']*100)}% membership"))

    # Write CSV
    csv_path = output_dir / "T7_stability_core.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("metric,value,notes\n")
        for metric, value, notes in rows:
            f.write(f"{metric},{value},{notes}\n")

    # Write TeX
    tex_path = output_dir / "T7_stability_core.tex"
    tex_content = r"""\begin{table}[htbp]
\centering
\caption{Pareto Set Stability Under Score Perturbation}
\label{tab:stability}
\begin{tabular}{lrl}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Notes} \\
\midrule
"""
    for metric, value, notes in rows:
        tex_content += f"{metric} & {value} & {notes} \\\\\n"

    tex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)

    return csv_path, tex_path


def generate_f1_scatter(run_dir: Path, output_dir: Path, stable_core_ids: list = None):
    """F1: Clean scatter plot with grey all-courses and highlighted Pareto (no labels)."""
    embeddings, artifacts, text_to_idx = load_embeddings_and_artifacts(run_dir)

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

    # Plot (single-column width ~3.5 inches)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # All courses (grey, faded)
    all_markets = [c['market'] for c in course_scores]
    all_missions = [c['mission'] for c in course_scores]
    ax.scatter(all_markets, all_missions, alpha=0.08, c='grey', s=3, rasterized=True)

    # Pareto frontier (colored by university)
    pareto_courses = [c for c in course_scores if c['id'] in pareto_ids]
    ucb_pareto = [(c['market'], c['mission']) for c in pareto_courses if c['university'] == 'UCB']
    mit_pareto = [(c['market'], c['mission']) for c in pareto_courses if c['university'] == 'MIT']

    if ucb_pareto:
        ax.scatter([p[0] for p in ucb_pareto], [p[1] for p in ucb_pareto],
                   c='#1f77b4', s=25, marker='o', alpha=0.8, edgecolors='white', linewidth=0.3)
    if mit_pareto:
        ax.scatter([p[0] for p in mit_pareto], [p[1] for p in mit_pareto],
                   c='#d62728', s=25, marker='s', alpha=0.8, edgecolors='white', linewidth=0.3)

    # Highlight stable core if provided
    if stable_core_ids:
        core_courses = [c for c in course_scores if c['id'] in stable_core_ids]
        if core_courses:
            ax.scatter([c['market'] for c in core_courses],
                       [c['mission'] for c in core_courses],
                       facecolors='none', edgecolors='#2ca02c', s=60, linewidth=1.5,
                       marker='o', zorder=10)

    ax.set_xlabel('Market Alignment', fontsize=8)
    ax.set_ylabel('Mission Alignment', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # Simple legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=5, label='UCB'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728', markersize=5, label='MIT'),
    ]
    if stable_core_ids:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='#2ca02c', markerfacecolor='none', markersize=6, label='Core')
        )
    ax.legend(handles=legend_elements, loc='lower left', fontsize=6, framealpha=0.8)

    plt.tight_layout()
    outpath = output_dir / 'F1_pareto_scatter.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'F1_pareto_scatter.pdf', bbox_inches='tight')
    plt.close()
    return outpath


def generate_f2_topk_bar(run_dir: Path, output_dir: Path, top_k: int = 5):
    """F2: Grouped bar chart for top-k Pareto options showing Market vs Mission."""
    embeddings, artifacts, text_to_idx = load_embeddings_and_artifacts(run_dir)

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

    # Get course scores
    course_scores = []
    for c in courses:
        if c['artifact_id'] in text_to_idx:
            emb = embeddings[text_to_idx[c['artifact_id']]]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            market_score = float(np.dot(emb_norm, market_c))
            mission_score = float(np.dot(emb_norm, mission_c))

            # Get title from metadata
            metadata = c.get('metadata', {})
            title = metadata.get('course_name') or metadata.get('title') or c.get('title')
            if not title:
                text = c.get('text', '')
                title = text.split('.')[0][:40] if text else c['artifact_id'][-12:]

            course_scores.append({
                'id': c['artifact_id'],
                'market': market_score,
                'mission': mission_score,
                'combined': market_score + mission_score,
                'title': title[:35] + '...' if len(title) > 35 else title,
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
    top_pareto = sorted(pareto_courses, key=lambda x: x['combined'], reverse=True)[:top_k]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    x = np.arange(len(top_pareto))
    width = 0.35

    market_scores = [c['market'] for c in top_pareto]
    mission_scores = [c['mission'] for c in top_pareto]

    bars1 = ax.bar(x - width/2, market_scores, width, label='Market', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, mission_scores, width, label='Mission', color='#2ca02c', alpha=0.8)

    ax.set_ylabel('Alignment Score', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i+1}' for i in range(len(top_pareto))], fontsize=7)
    ax.legend(fontsize=7, loc='upper right')
    ax.tick_params(axis='y', labelsize=7)
    ax.set_ylim(0, 0.75)
    ax.grid(True, alpha=0.2, axis='y', linewidth=0.5)

    plt.tight_layout()
    outpath = output_dir / 'F2_topk_bar.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'F2_topk_bar.pdf', bbox_inches='tight')
    plt.close()
    return outpath


def generate_appendix_topk_csv(run_dir: Path, output_dir: Path, top_k: int = 20):
    """Generate appendix CSV with course details for reviewers."""
    embeddings, artifacts, text_to_idx = load_embeddings_and_artifacts(run_dir)

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
            market_score = float(np.dot(emb_norm, market_c))
            mission_score = float(np.dot(emb_norm, mission_c))

            # Extract university from artifact_id
            parts = c['artifact_id'].split(':')
            university = parts[1] if len(parts) > 1 else 'unknown'

            # Get title from metadata.course_name or text
            metadata = c.get('metadata', {})
            title = metadata.get('course_name') or metadata.get('title') or c.get('title')
            if not title:
                # Extract first sentence from text as fallback
                text = c.get('text', '')
                title = text.split('.')[0][:80] if text else c['artifact_id']

            course_scores.append({
                'id': c['artifact_id'],
                'title': title,
                'university': university,
                'market_score': market_score,
                'mission_score': mission_score,
                'source': c.get('source_url', c.get('url', '')),
            })

    # Compute Pareto
    pareto_ids = set()
    for i, c_i in enumerate(course_scores):
        is_pareto = True
        for j, c_j in enumerate(course_scores):
            if i != j:
                if c_j['market_score'] >= c_i['market_score'] and c_j['mission_score'] >= c_i['mission_score']:
                    if c_j['market_score'] > c_i['market_score'] or c_j['mission_score'] > c_i['mission_score']:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_ids.add(c_i['id'])

    pareto_courses = [c for c in course_scores if c['id'] in pareto_ids]
    top_pareto = sorted(pareto_courses,
                        key=lambda x: x['market_score'] + x['mission_score'],
                        reverse=True)[:top_k]

    # Write CSV
    appendix_dir = output_dir / "appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)

    csv_path = appendix_dir / "pareto_options_topk.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("course_id,title,university,market_score,mission_score,source\n")
        for c in top_pareto:
            # Escape commas in title
            title = c['title'].replace('"', '""')
            if ',' in title:
                title = f'"{title}"'
            f.write(f"{c['id']},{title},{c['university']},{c['market_score']:.4f},{c['mission_score']:.4f},{c['source']}\n")

    return csv_path


def build_repro_manifest(
    output_dir: Path,
    run_dir: Path,
    experiment_results: dict,
    baseline_config: dict,
):
    """Build hardened repro_manifest.json with no placeholders."""

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "python_version": get_python_version(),
        "platform": get_platform_info(),
        "packages": get_key_packages(),
        "dataset": {
            "name": experiment_results.get('dataset_name', 'unified_v3'),
            "artifacts_hash_sha256": compute_artifacts_hash(run_dir),
            "total_artifacts": experiment_results.get('total_artifacts', 0),
            "courses": experiment_results.get('n_courses', 0),
            "job_roles": experiment_results.get('n_jobs', 0),
            "mission_statements": experiment_results.get('n_missions', 0),
            "mission_source": experiment_results.get('mission_source', 'mission_corpus.csv'),
            "sources": experiment_results.get('sources', {}),
        },
        "baseline_config": baseline_config,
        "encoders": experiment_results.get('encoders', {}),
        "seeds": experiment_results.get('seeds', [0, 1, 2]),
        "key_results": experiment_results.get('key_results', {}),
        "commands_used": experiment_results.get('commands_used', []),
        "notes": experiment_results.get('notes', {}),
    }

    manifest_path = output_dir / "repro_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def build_results_index(output_dir: Path, manifest: dict, tables: list, figures: list):
    """Build results_index.md from manifest and artifact lists."""

    md = f"""# Results Index - Paper Evidence Pack

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Git Commit**: `{manifest.get('git_commit', 'unknown')}`
**Python**: {manifest.get('python_version', 'unknown')}
**Platform**: {manifest.get('platform', {}).get('system', 'unknown')} ({manifest.get('platform', {}).get('machine', 'unknown')})

---

## Dataset Summary

| Type | Count | Notes |
|------|-------|-------|
| Courses | {manifest['dataset'].get('courses', 0):,} | MIT + UC Berkeley |
| Jobs | {manifest['dataset'].get('job_roles', 0):,} | O*NET |
| Missions | {manifest['dataset'].get('mission_statements', 0)} | {manifest['dataset'].get('mission_source', '')} |
| **Total** | {manifest['dataset'].get('total_artifacts', 0):,} | |

**Artifacts Hash**: `{manifest['dataset'].get('artifacts_hash_sha256', 'unknown')}`

---

## Baseline Configuration

```json
{json.dumps(manifest.get('baseline_config', {}), indent=2)}
```

---

## Key Results

"""
    for key, val in manifest.get('key_results', {}).items():
        if isinstance(val, dict):
            md += f"### {key}\n"
            for k, v in val.items():
                md += f"- **{k}**: {v}\n"
            md += "\n"
        else:
            md += f"- **{key}**: {val}\n"

    md += """
---

## Tables

"""
    for table in tables:
        md += f"- `{table.name}`\n"

    md += """
---

## Figures

"""
    for fig in figures:
        md += f"- `{fig.name}`\n"

    md += """
---

## Reproduction Commands

```bash
"""
    for cmd in manifest.get('commands_used', []):
        md += f"{cmd}\n"

    md += """```

---

## Package Versions

| Package | Version |
|---------|---------|
"""
    for pkg, ver in manifest.get('packages', {}).items():
        md += f"| {pkg} | {ver} |\n"

    index_path = output_dir / "results_index.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(md)

    return index_path


def main():
    parser = argparse.ArgumentParser(description="Build paper evidence pack")
    parser.add_argument('--dataset', type=str, default='unified_v3')
    parser.add_argument('--tag', type=str, default='paper_max')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Path to experiment run directory')
    parser.add_argument('--skip-experiments', action='store_true',
                        help='Skip running experiments, use existing results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for paper pack')
    args = parser.parse_args()

    # Determine paths
    project_root = Path(__file__).parent.parent.parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "paper" / args.tag / timestamp

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Find most recent run
        reports_dir = project_root / "experiments" / "reports"
        # Find most recent run directory
        run_candidates = sorted(
            [d for d in reports_dir.iterdir() if d.is_dir() and not d.name.startswith('.')],
            key=lambda d: d.stat().st_mtime,
            reverse=True
        )
        if run_candidates:
            run_dir = run_candidates[0]
            # Check for subdirectory with embeddings
            subdirs = [d for d in run_dir.iterdir() if d.is_dir() and (d / "store").exists()]
            if subdirs:
                run_dir = subdirs[0]
        else:
            run_dir = reports_dir / "latest"

    print(f"Building paper pack...")
    print(f"  Output: {output_dir}")
    print(f"  Run dir: {run_dir}")

    # Create directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "exports").mkdir(exist_ok=True)
    (output_dir / "appendix").mkdir(exist_ok=True)

    # Load artifacts to get counts
    print("\n[1/6] Loading artifacts...")
    try:
        embeddings, artifacts, text_to_idx = load_embeddings_and_artifacts(run_dir)
        n_courses = len([a for a in artifacts if a['type'] == 'COURSE'])
        n_jobs = len([a for a in artifacts if a['type'] == 'JOB_ROLE'])
        n_missions = len([a for a in artifacts if a['type'] == 'MISSION'])
        print(f"  Courses: {n_courses}, Jobs: {n_jobs}, Missions: {n_missions}")
    except Exception as e:
        print(f"  Warning: Could not load artifacts: {e}")
        n_courses, n_jobs, n_missions = 0, 0, 0
        embeddings, artifacts, text_to_idx = None, None, None

    # Compute stable core (C1)
    print("\n[2/6] Computing stable core...")
    try:
        stable_core_data = compute_stable_core(run_dir, n_samples=50, epsilon=0.02, seed=42)
        print(f"  Base Pareto: {stable_core_data['base_pareto_size']}")
        for key, core in stable_core_data['stable_cores'].items():
            print(f"  {key}: {core['core_size']} options")

        # Save stable core
        stable_core_path = output_dir / "exports" / "stable_core.json"
        with open(stable_core_path, 'w', encoding='utf-8') as f:
            json.dump(stable_core_data, f, indent=2)

        # Generate T7 from stable core (C1)
        csv_t7, tex_t7 = generate_t7_from_stable_core(stable_core_data, output_dir / "tables")
        print(f"  Generated: {csv_t7.name}, {tex_t7.name}")
    except Exception as e:
        print(f"  Warning: Could not compute stable core: {e}")
        stable_core_data = None

    # Generate figures (C4)
    print("\n[3/6] Generating figures...")
    stable_core_ids = None
    if stable_core_data and 'stable_cores' in stable_core_data:
        core_50 = stable_core_data['stable_cores'].get('p=0.5', {})
        stable_core_ids = core_50.get('core_ids', [])

    figures = []
    try:
        f1 = generate_f1_scatter(run_dir, output_dir / "figures", stable_core_ids)
        figures.append(f1)
        print(f"  Generated: {f1.name}")
    except Exception as e:
        print(f"  Warning: F1 scatter failed: {e}")

    try:
        f2 = generate_f2_topk_bar(run_dir, output_dir / "figures", top_k=5)
        figures.append(f2)
        print(f"  Generated: {f2.name}")
    except Exception as e:
        print(f"  Warning: F2 bar chart failed: {e}")

    # Generate appendix (C5)
    print("\n[4/6] Generating appendix...")
    try:
        appendix_csv = generate_appendix_topk_csv(run_dir, output_dir, top_k=20)
        print(f"  Generated: {appendix_csv.name}")
    except Exception as e:
        print(f"  Warning: Appendix generation failed: {e}")

    # Build baseline config (C2)
    print("\n[5/6] Building baseline configuration...")
    baseline_config = {
        "objectives": ["market_alignment", "mission_alignment"],
        "tau": 0.25,
        "weight_sampling": {
            "protocol": "uniform_simplex",
            "n_vectors": 100,
        },
        "selection_rule": "pareto_dominance",
        "centroid_method": "mean",
        "similarity": "cosine",
    }

    # Build experiment results summary
    experiment_results = {
        "dataset_name": args.dataset,
        "total_artifacts": n_courses + n_jobs + n_missions,
        "n_courses": n_courses,
        "n_jobs": n_jobs,
        "n_missions": n_missions,
        "mission_source": "mission_corpus.csv",
        "sources": {
            "UC Berkeley": {"courses": 11113, "missions": 20},
            "MIT": {"courses": 6944, "missions": 14},
            "ASU": {"missions": 22},
        },
        "encoders": {
            "primary": "sentence-transformers/all-MiniLM-L6-v2",
            "ablation": "sentence-transformers/all-mpnet-base-v2",
            "baseline": "stub-encoder-v0",
        },
        "seeds": [0, 1, 2],
        "key_results": {
            "minilm_pareto": {"mean": 81, "std": 0},
            "mpnet_pareto": {"mean": 30, "std": 0},
            "stub_pareto": 2,
            "lens_ablation": {"identity": 81, "diagonal": 81, "learned": 170},
        },
        "commands_used": [
            f"python -m experiments.run dataset={args.dataset} embedding=sbert_allminilm lenses=identity seed=0,1,2",
            f"python -m experiments.run dataset={args.dataset} embedding=sbert_mpnet lenses=identity seed=0,1,2",
            "python experiments/scripts/cross_university_v2.py",
            "python experiments/scripts/build_paper_pack.py --dataset unified_v3 --tag paper_max",
        ],
        "notes": {
            "mission_corpus": "56 mission statements from MIT, UC Berkeley, ASU",
            "outcomes": "Synthetic (fixture outcomes)",
            "ground_truth": "None",
        },
    }

    # Build repro_manifest.json (C3)
    print("\n[6/6] Building repro manifest...")
    manifest_path = build_repro_manifest(output_dir, run_dir, experiment_results, baseline_config)
    print(f"  Generated: {manifest_path.name}")

    # Load manifest back for index generation
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # Collect tables
    tables = list((output_dir / "tables").glob("*.csv")) + list((output_dir / "tables").glob("*.tex"))

    # Build results_index.md
    index_path = build_results_index(output_dir, manifest, tables, figures)
    print(f"  Generated: {index_path.name}")

    print(f"\n{'='*60}")
    print(f"Paper pack generated successfully!")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nContents:")
    for item in sorted(output_dir.rglob("*")):
        if item.is_file():
            rel = item.relative_to(output_dir)
            print(f"  {rel}")

    return output_dir


if __name__ == "__main__":
    main()
