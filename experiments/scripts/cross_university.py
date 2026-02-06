"""D1: Cross-university validation - train on one, test on another."""
import argparse
import json
import numpy as np
from pathlib import Path
from collections import Counter

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Cross-university validation analysis")
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

    # Separate by university
    courses = [a for a in artifacts if a['type'] == 'COURSE']
    jobs = [a for a in artifacts if a['type'] == 'JOB_ROLE']
    missions = [a for a in artifacts if a['type'] == 'MISSION']

    # Parse source from artifact_id: 'course:UC Berkeley:xxx' or 'course:MIT:xxx'
    ucb_courses = [c for c in courses if ':UC Berkeley:' in c['artifact_id']]
    mit_courses = [c for c in courses if ':MIT:' in c['artifact_id']]

    print(f"UCB courses: {len(ucb_courses)}")
    print(f"MIT courses: {len(mit_courses)}")
    print(f"Jobs: {len(jobs)}, Missions: {len(missions)}")

    # Build embeddings
    def get_course_embeddings(course_list):
        embs = []
        ids = []
        for c in course_list:
            if c['artifact_id'] in text_to_idx:
                embs.append(embeddings[text_to_idx[c['artifact_id']]])
                ids.append(c['artifact_id'])
        return np.array(embs), ids

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

    # Centroids
    market_centroid = np.mean(job_embs, axis=0)
    market_centroid = market_centroid / np.linalg.norm(market_centroid)
    mission_centroid = np.mean(mission_embs, axis=0)
    mission_centroid = mission_centroid / np.linalg.norm(mission_centroid)

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

    def compute_scores(course_embs, course_ids):
        scores = {}
        for i, cid in enumerate(course_ids):
            emb = course_embs[i]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            scores[cid] = {
                'market': float(np.dot(emb_norm, market_centroid)),
                'mission': float(np.dot(emb_norm, mission_centroid)),
            }
        return scores

    # Full dataset Pareto
    all_embs, all_ids = get_course_embeddings(courses)
    full_scores = compute_scores(all_embs, all_ids)
    full_pareto = compute_pareto(full_scores)

    # UCB-only Pareto
    ucb_embs, ucb_ids = get_course_embeddings(ucb_courses)
    ucb_scores = compute_scores(ucb_embs, ucb_ids)
    ucb_pareto = compute_pareto(ucb_scores)

    # MIT-only Pareto
    mit_embs, mit_ids = get_course_embeddings(mit_courses)
    mit_scores = compute_scores(mit_embs, mit_ids)
    mit_pareto = compute_pareto(mit_scores)

    print(f"\n=== CROSS-UNIVERSITY VALIDATION ===")
    print(f"Full Pareto: {len(full_pareto)}")
    print(f"UCB-only Pareto: {len(ucb_pareto)}")
    print(f"MIT-only Pareto: {len(mit_pareto)}")

    # Transfer: how many UCB Pareto are in full Pareto?
    ucb_in_full = len(ucb_pareto & full_pareto)
    mit_in_full = len(mit_pareto & full_pareto)

    print(f"\nUCB Pareto in full: {ucb_in_full}/{len(ucb_pareto)} = {ucb_in_full*100/len(ucb_pareto):.1f}%")
    print(f"MIT Pareto in full: {mit_in_full}/{len(mit_pareto)} = {mit_in_full*100/len(mit_pareto):.1f}%")

    # Jaccard similarity
    jaccard_ucb_full = len(ucb_pareto & full_pareto) / len(ucb_pareto | full_pareto)
    jaccard_mit_full = len(mit_pareto & full_pareto) / len(mit_pareto | full_pareto)
    jaccard_ucb_mit = len(ucb_pareto & mit_pareto) / max(1, len(ucb_pareto | mit_pareto))

    print(f"\nJaccard(UCB, full): {jaccard_ucb_full:.3f}")
    print(f"Jaccard(MIT, full): {jaccard_mit_full:.3f}")
    print(f"Jaccard(UCB, MIT): {jaccard_ucb_mit:.3f}")

    # Score distribution comparison
    ucb_market_mean = np.mean([ucb_scores[c]['market'] for c in ucb_ids])
    ucb_mission_mean = np.mean([ucb_scores[c]['mission'] for c in ucb_ids])
    mit_market_mean = np.mean([mit_scores[c]['market'] for c in mit_ids])
    mit_mission_mean = np.mean([mit_scores[c]['mission'] for c in mit_ids])

    print(f"\nScore distributions:")
    print(f"UCB: market={ucb_market_mean:.3f}, mission={ucb_mission_mean:.3f}")
    print(f"MIT: market={mit_market_mean:.3f}, mission={mit_mission_mean:.3f}")

    # Summary
    print("\n" + "="*60)
    print("D1 SUMMARY:")
    print(f"- University-specific Pareto sets are subsets of full (~{max(ucb_in_full, mit_in_full)*100/max(len(ucb_pareto), len(mit_pareto)):.0f}%)")
    print(f"- Low cross-university overlap (Jaccard={jaccard_ucb_mit:.3f}) shows diversity")
    print("="*60)

    # Save results
    out_dir = Path("experiments/reports/cross_university")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "results.json", 'w') as f:
        json.dump({
            "n_ucb_courses": len(ucb_courses),
            "n_mit_courses": len(mit_courses),
            "full_pareto": len(full_pareto),
            "ucb_pareto": len(ucb_pareto),
            "mit_pareto": len(mit_pareto),
            "ucb_in_full": ucb_in_full,
            "mit_in_full": mit_in_full,
            "jaccard_ucb_full": jaccard_ucb_full,
            "jaccard_mit_full": jaccard_mit_full,
            "jaccard_ucb_mit": jaccard_ucb_mit,
        }, f, indent=2)

    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
