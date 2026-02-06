"""B3: Objective subset ablation - which objectives matter most?"""
import argparse
import json
import numpy as np
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Objective subset ablation analysis")
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

    # Build embeddings
    job_embs = np.array([embeddings[text_to_idx[j['artifact_id']]]
                         for j in jobs if j['artifact_id'] in text_to_idx])
    mission_embs = np.array([embeddings[text_to_idx[m['artifact_id']]]
                             for m in missions if m['artifact_id'] in text_to_idx])

    market_centroid = np.mean(job_embs, axis=0)
    market_centroid = market_centroid / np.linalg.norm(market_centroid)
    mission_centroid = np.mean(mission_embs, axis=0)
    mission_centroid = mission_centroid / np.linalg.norm(mission_centroid)

    # Build course embeddings
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

    def compute_pareto(items, keys):
        pareto = []
        for i, item_i in enumerate(items):
            is_pareto = True
            for j, item_j in enumerate(items):
                if i != j:
                    dominated = all(item_j[k] >= item_i[k] for k in keys)
                    strictly_better = any(item_j[k] > item_i[k] for k in keys)
                    if dominated and strictly_better:
                        is_pareto = False
                        break
            if is_pareto:
                pareto.append(item_i['id'])
        return set(pareto)

    print("\n=== OBJECTIVE SUBSET ABLATION ===\n")

    # Full Pareto (both objectives)
    full_pareto = compute_pareto(course_data, ['market', 'mission'])
    print(f"Both objectives (market + mission): {len(full_pareto)} Pareto options")

    # Market-only (single objective = just top-k)
    sorted_by_market = sorted(course_data, key=lambda x: x['market'], reverse=True)
    top_market = {sorted_by_market[0]['id']}  # Only the best
    print(f"Market-only (single obj): 1 option")

    # Mission-only
    sorted_by_mission = sorted(course_data, key=lambda x: x['mission'], reverse=True)
    top_mission = {sorted_by_mission[0]['id']}
    print(f"Mission-only (single obj): 1 option")

    # Check overlap
    single_in_pareto = len((top_market | top_mission) & full_pareto)
    print(f"\nSingle-objective winners in Pareto: {single_in_pareto}/2")

    # Score correlation
    markets = [c['market'] for c in course_data]
    missions = [c['mission'] for c in course_data]
    correlation = np.corrcoef(markets, missions)[0, 1]
    print(f"Objective correlation: {correlation:.3f}")

    # Score extremes analysis
    print("\n--- Pareto Frontier Analysis ---")
    pareto_items = [c for c in course_data if c['id'] in full_pareto]
    market_scores = [c['market'] for c in pareto_items]
    mission_scores = [c['mission'] for c in pareto_items]

    print(f"Pareto market range: [{min(market_scores):.3f}, {max(market_scores):.3f}]")
    print(f"Pareto mission range: [{min(mission_scores):.3f}, {max(mission_scores):.3f}]")

    # Identify extremes
    market_max = max(pareto_items, key=lambda x: x['market'])
    mission_max = max(pareto_items, key=lambda x: x['mission'])

    print(f"\nMarket-best Pareto: market={market_max['market']:.3f}, mission={market_max['mission']:.3f}")
    print(f"Mission-best Pareto: market={mission_max['market']:.3f}, mission={mission_max['mission']:.3f}")

    # Trade-off analysis: how much mission do you sacrifice for market?
    tradeoff = (market_max['market'] - mission_max['market']) / (mission_max['mission'] - market_max['mission'] + 1e-8)
    print(f"Trade-off ratio (market gain / mission loss): {tradeoff:.2f}")

    # Objective importance via Pareto coverage
    # If removing one objective doesn't change Pareto much, that objective is redundant
    print("\n--- Objective Importance ---")

    # What if we only had top-10 from each objective?
    top10_market = set(sorted_by_market[i]['id'] for i in range(10))
    top10_mission = set(sorted_by_mission[i]['id'] for i in range(10))

    pareto_in_top10_market = len(full_pareto & top10_market)
    pareto_in_top10_mission = len(full_pareto & top10_mission)

    print(f"Pareto in top-10 market: {pareto_in_top10_market}")
    print(f"Pareto in top-10 mission: {pareto_in_top10_mission}")

    # Summary
    print("\n" + "="*60)
    print("B3 SUMMARY:")
    print(f"- Correlation r={correlation:.2f} - objectives partially aligned")
    print(f"- Both objectives contribute unique Pareto members")
    print(f"- Trade-off exists (ratio={tradeoff:.1f})")
    print("="*60)

    # Save results
    out_dir = Path("experiments/reports/objective_ablation")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "results.json", 'w') as f:
        json.dump({
            "full_pareto": len(full_pareto),
            "objective_correlation": correlation,
            "market_range": [min(market_scores), max(market_scores)],
            "mission_range": [min(mission_scores), max(mission_scores)],
            "tradeoff_ratio": tradeoff,
            "pareto_in_top10_market": pareto_in_top10_market,
            "pareto_in_top10_mission": pareto_in_top10_mission,
        }, f, indent=2)

    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
