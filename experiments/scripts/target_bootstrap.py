"""C3: Target Corpus Bootstrap Stability - test sensitivity to target sampling."""
import argparse
import json
import numpy as np
from pathlib import Path
from collections import Counter

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Target corpus bootstrap stability analysis")
    parser.add_argument('--run-dir', type=Path,
                        default=_REPO_ROOT / 'experiments/reports/latest',
                        help='Path to experiment run directory')
    args = parser.parse_args()

    run_dir = args.run_dir

    # Load data
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

    print(f"Courses: {len(courses)}, Jobs: {len(jobs)}, Missions: {len(missions)}")

    # Get embeddings
    job_embs = []
    for j in jobs:
        if j['artifact_id'] in text_to_idx:
            job_embs.append(embeddings[text_to_idx[j['artifact_id']]])
    job_embs = np.array(job_embs)

    mission_embs = []
    for m in missions:
        if m['artifact_id'] in text_to_idx:
            mission_embs.append(embeddings[text_to_idx[m['artifact_id']]])
    mission_embs = np.array(mission_embs)

    print(f"Job embeddings: {job_embs.shape}, Mission embeddings: {mission_embs.shape}")

    def compute_pareto(scores):
        keys = ['market', 'mission']
        items = list(scores.items())
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

    def compute_scores_with_centroids(market_c, mission_c):
        scores = {}
        for c in courses:
            if c['artifact_id'] in text_to_idx:
                idx = text_to_idx[c['artifact_id']]
                emb = embeddings[idx]
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                scores[c['artifact_id']] = {
                    'market': float(np.dot(emb_norm, market_c)),
                    'mission': float(np.dot(emb_norm, mission_c)),
                }
        return scores

    # Base centroids
    base_market = np.mean(job_embs, axis=0)
    base_market = base_market / np.linalg.norm(base_market)
    base_mission = np.mean(mission_embs, axis=0)
    base_mission = base_mission / np.linalg.norm(base_mission)

    base_scores = compute_scores_with_centroids(base_market, base_mission)
    base_pareto = compute_pareto(base_scores)
    print(f"\nBase Pareto: {len(base_pareto)} options")

    # Bootstrap experiment
    np.random.seed(42)
    n_bootstrap = 50

    print("\n--- MARKET TARGET BOOTSTRAP ---")
    membership_market = Counter()
    pareto_sizes_market = []

    for i in range(n_bootstrap):
        idx = np.random.choice(len(job_embs), len(job_embs), replace=True)
        boot_market = np.mean(job_embs[idx], axis=0)
        boot_market = boot_market / np.linalg.norm(boot_market)

        scores = compute_scores_with_centroids(boot_market, base_mission)
        pareto = compute_pareto(scores)
        pareto_sizes_market.append(len(pareto))

        for opt in pareto:
            membership_market[opt] += 1

    print(f"Pareto size: {np.mean(pareto_sizes_market):.1f} +/- {np.std(pareto_sizes_market):.1f}")
    base_freqs_market = [membership_market.get(opt, 0)*100/n_bootstrap for opt in base_pareto]
    print(f"Base Pareto members in >50%: {sum(1 for f in base_freqs_market if f >= 50)}/{len(base_pareto)}")

    # Combined bootstrap
    print("\n--- COMBINED BOOTSTRAP ---")
    membership_combined = Counter()
    pareto_sizes_combined = []

    for i in range(n_bootstrap):
        idx_job = np.random.choice(len(job_embs), len(job_embs), replace=True)
        idx_mission = np.random.choice(len(mission_embs), len(mission_embs), replace=True)

        boot_market = np.mean(job_embs[idx_job], axis=0)
        boot_market = boot_market / np.linalg.norm(boot_market)
        boot_mission = np.mean(mission_embs[idx_mission], axis=0)
        boot_mission = boot_mission / np.linalg.norm(boot_mission)

        scores = compute_scores_with_centroids(boot_market, boot_mission)
        pareto = compute_pareto(scores)
        pareto_sizes_combined.append(len(pareto))

        for opt in pareto:
            membership_combined[opt] += 1

    print(f"Pareto size: {np.mean(pareto_sizes_combined):.1f} +/- {np.std(pareto_sizes_combined):.1f}")
    base_freqs_combined = [membership_combined.get(opt, 0)*100/n_bootstrap for opt in base_pareto]
    stable_core = sum(1 for f in base_freqs_combined if f >= 50)
    print(f"Stable core (>50%): {stable_core}/{len(base_pareto)}")

    print("\n" + "="*60)
    print("C3 SUMMARY: Mission corpus too small (n=3) - recommend expansion")
    print("="*60)

    # Save results
    out_dir = Path("experiments/reports/stability")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "target_bootstrap.json", 'w') as f:
        json.dump({
            "n_bootstrap": n_bootstrap,
            "n_jobs": len(job_embs),
            "n_missions": len(mission_embs),
            "base_pareto_size": len(base_pareto),
            "market_bootstrap_stable": sum(1 for f in base_freqs_market if f >= 50),
            "combined_stable_core": stable_core,
        }, f, indent=2)

if __name__ == "__main__":
    main()
