"""
Stable Core from Perturbations - compute membership frequencies and stable core.

This script:
1. Loads embeddings and artifacts
2. Runs N perturbation samples at given epsilon
3. Tracks which options appear in Pareto across samples
4. Computes stable core at threshold p (e.g., p=0.5 means >50% membership)
5. Saves results to paper artifacts
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse

_REPO_ROOT = Path(__file__).resolve().parents[2]

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default=str(_REPO_ROOT / 'experiments/reports/latest'))
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=0.02)
    parser.add_argument('--thresholds', type=str, default='0.5,0.8')
    parser.add_argument('--output_dir', type=str,
                        default=str(_REPO_ROOT / 'experiments/reports/stability'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [float(t) for t in args.thresholds.split(',')]

    print(f"Loading data from {run_dir}...")

    # Load embeddings
    cache_dir = run_dir / "cache_embeddings" / "sbert_sentence-transformers_all-MiniLM-L6-v2"
    embeddings = np.load(cache_dir / "vecs.npy")
    with open(cache_dir / "index.json", encoding='utf-8') as f:
        index = json.load(f)

    # Load artifacts
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

    print(f"Courses: {len(courses)}, Jobs: {len(jobs)}, Missions: {len(missions)}")

    # Build embeddings
    job_embs = np.array([embeddings[text_to_idx[j['artifact_id']]]
                         for j in jobs if j['artifact_id'] in text_to_idx])
    mission_embs = np.array([embeddings[text_to_idx[m['artifact_id']]]
                             for m in missions if m['artifact_id'] in text_to_idx])

    # Base centroids
    market_centroid = np.mean(job_embs, axis=0)
    market_centroid = market_centroid / np.linalg.norm(market_centroid)
    mission_centroid = np.mean(mission_embs, axis=0)
    mission_centroid = mission_centroid / np.linalg.norm(mission_centroid)

    # Build course scores
    course_data = []
    for c in courses:
        if c['artifact_id'] in text_to_idx:
            emb = embeddings[text_to_idx[c['artifact_id']]]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            course_data.append({
                'id': c['artifact_id'],
                'market': float(np.dot(emb_norm, market_centroid)),
                'mission': float(np.dot(emb_norm, mission_centroid)),
            })

    # Base Pareto
    base_scores = {c['id']: {'market': c['market'], 'mission': c['mission']}
                   for c in course_data}
    base_pareto = compute_pareto(base_scores, ['market', 'mission'])
    print(f"\nBase Pareto size: {len(base_pareto)}")

    # Perturbation sampling
    np.random.seed(args.seed)
    membership = Counter()
    pareto_sizes = []

    print(f"\nRunning {args.n_samples} perturbation samples with eps={args.epsilon}...")

    for i in range(args.n_samples):
        if (i + 1) % 20 == 0:
            print(f"  Sample {i+1}/{args.n_samples}")

        # Perturb scores
        perturbed_scores = {}
        for c in course_data:
            perturbed_scores[c['id']] = {
                'market': c['market'] + np.random.uniform(-args.epsilon, args.epsilon),
                'mission': c['mission'] + np.random.uniform(-args.epsilon, args.epsilon),
            }

        # Compute Pareto
        pareto = compute_pareto(perturbed_scores, ['market', 'mission'])
        pareto_sizes.append(len(pareto))

        for opt in pareto:
            membership[opt] += 1

    # Compute stable core at different thresholds
    stable_cores = {}
    for p in thresholds:
        min_count = int(args.n_samples * p)
        core = [opt for opt, count in membership.items() if count >= min_count]
        stable_cores[f'p={p}'] = {
            'threshold': p,
            'min_count': min_count,
            'core_size': len(core),
            'core_ids': sorted(core)
        }

    # Summary stats
    mean_pareto_size = np.mean(pareto_sizes)
    std_pareto_size = np.std(pareto_sizes)

    # Top options by frequency
    top_options = membership.most_common(20)

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'run_dir': str(run_dir),
            'n_samples': args.n_samples,
            'epsilon': args.epsilon,
            'seed': args.seed,
        },
        'base_pareto_size': len(base_pareto),
        'perturbed_pareto_size': {
            'mean': mean_pareto_size,
            'std': std_pareto_size,
            'min': min(pareto_sizes),
            'max': max(pareto_sizes),
        },
        'stable_cores': stable_cores,
        'top_options': [{'id': opt, 'frequency': count, 'pct': count * 100 / args.n_samples}
                        for opt, count in top_options],
        'full_membership': {opt: count for opt, count in membership.items()},
    }

    # Save results
    output_file = output_dir / 'stable_core.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("\n=== RESULTS ===")
    print(f"Base Pareto: {len(base_pareto)}")
    print(f"Perturbed Pareto: {mean_pareto_size:.1f} +/- {std_pareto_size:.1f}")
    print("\nStable Cores:")
    for key, core in stable_cores.items():
        print(f"  {key}: {core['core_size']} options (>= {core['min_count']} appearances)")

    print("\nTop 10 most stable options:")
    for opt, count in top_options[:10]:
        print(f"  {opt[:40]}: {count}/{args.n_samples} ({count*100/args.n_samples:.1f}%)")

    print(f"\nResults saved to {output_file}")
    return results

if __name__ == "__main__":
    main()
