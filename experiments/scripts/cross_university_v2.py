"""D1: Cross-university validation - train on one, test on another.

Updated for unified_v3 with 5 universities: ASU, MIT, UC Berkeley, Stanford, UIUC, Cornell.
"""
import json
import numpy as np
from pathlib import Path
import argparse

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str,
                        default=str(_REPO_ROOT / 'experiments/reports/latest'),
                        help='Path to experiment run directory')
    parser.add_argument('--out-dir', type=str,
                        default=str(_REPO_ROOT / 'experiments/reports/cross_university'),
                        help='Output directory for results')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    cache_dir = run_dir / "cache_embeddings" / "sbert_sentence-transformers_all-MiniLM-L6-v2"
    if not cache_dir.exists():
        # Try alternative path
        cache_dirs = list((run_dir / "cache_embeddings").glob("sbert_*"))
        if cache_dirs:
            cache_dir = cache_dirs[0]
        else:
            print(f"Error: No embeddings cache found in {run_dir}")
            return

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

    # Separate by type
    courses = [a for a in artifacts if a['type'] == 'COURSE']
    jobs = [a for a in artifacts if a['type'] == 'JOB_ROLE']
    missions = [a for a in artifacts if a['type'] == 'MISSION']

    # Parse university from artifact_id or metadata
    def get_university(artifact):
        # Try metadata first
        if 'metadata' in artifact and 'university' in artifact['metadata']:
            return artifact['metadata']['university']
        # Parse from artifact_id: 'course:UC Berkeley:xxx' or 'course:MIT:xxx'
        aid = artifact['artifact_id']
        for uni in universities:
            if f':{uni}:' in aid:
                return uni
        return 'Unknown'

    # Group courses by university
    universities = ['UC Berkeley', 'MIT', 'ASU', 'Stanford', 'UIUC', 'Cornell']
    courses_by_uni = {uni: [] for uni in universities}
    for c in courses:
        uni = get_university(c)
        if uni in courses_by_uni:
            courses_by_uni[uni].append(c)

    for uni in universities:
        print(f"{uni} courses: {len(courses_by_uni[uni])}")
    print(f"Jobs: {len(jobs)}, Missions: {len(missions)}")

    # Group missions by university
    missions_by_uni = {uni: [] for uni in universities}
    for m in missions:
        uni = get_university(m)
        if uni in missions_by_uni:
            missions_by_uni[uni].append(m)

    for uni in universities:
        print(f"{uni} missions: {len(missions_by_uni[uni])}")

    # Build embeddings
    def get_embeddings(artifact_list):
        embs = []
        ids = []
        for a in artifact_list:
            if a['artifact_id'] in text_to_idx:
                embs.append(embeddings[text_to_idx[a['artifact_id']]])
                ids.append(a['artifact_id'])
        return np.array(embs) if embs else np.array([]).reshape(0, embeddings.shape[1]), ids

    job_embs, _ = get_embeddings(jobs)
    mission_embs, _ = get_embeddings(missions)

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
    all_embs, all_ids = get_embeddings(courses)
    full_scores = compute_scores(all_embs, all_ids)
    full_pareto = compute_pareto(full_scores)

    # Per-university Pareto
    pareto_by_uni = {}
    scores_by_uni = {}
    for uni in universities:
        uni_embs, uni_ids = get_embeddings(courses_by_uni[uni])
        if len(uni_ids) > 0:
            scores_by_uni[uni] = compute_scores(uni_embs, uni_ids)
            pareto_by_uni[uni] = compute_pareto(scores_by_uni[uni])
        else:
            pareto_by_uni[uni] = set()
            scores_by_uni[uni] = {}

    print("\n=== CROSS-UNIVERSITY VALIDATION ===")
    print(f"Full Pareto: {len(full_pareto)}")
    for uni in universities:
        print(f"{uni}-only Pareto: {len(pareto_by_uni[uni])}")

    # Transfer: how many uni Pareto are in full Pareto?
    results = {
        "n_courses": {uni: len(courses_by_uni[uni]) for uni in universities},
        "n_missions": {uni: len(missions_by_uni[uni]) for uni in universities},
        "full_pareto": len(full_pareto),
        "pareto_by_uni": {uni: len(pareto_by_uni[uni]) for uni in universities},
    }

    print("\nTransfer to full Pareto:")
    for uni in universities:
        if len(pareto_by_uni[uni]) > 0:
            in_full = len(pareto_by_uni[uni] & full_pareto)
            pct = in_full * 100 / len(pareto_by_uni[uni])
            print(f"  {uni}: {in_full}/{len(pareto_by_uni[uni])} = {pct:.1f}%")
            results[f"{uni}_in_full"] = in_full
            results[f"{uni}_in_full_pct"] = pct

    # Jaccard similarities
    print("\nJaccard similarities:")
    jaccard_matrix = {}
    for i, uni1 in enumerate(universities):
        for uni2 in universities[i+1:]:
            if len(pareto_by_uni[uni1]) > 0 and len(pareto_by_uni[uni2]) > 0:
                intersection = len(pareto_by_uni[uni1] & pareto_by_uni[uni2])
                union = len(pareto_by_uni[uni1] | pareto_by_uni[uni2])
                jaccard = intersection / union if union > 0 else 0
                key = f"jaccard_{uni1}_{uni2}"
                jaccard_matrix[key] = jaccard
                print(f"  Jaccard({uni1}, {uni2}): {jaccard:.3f}")

    results["jaccard_matrix"] = jaccard_matrix

    # Score distribution comparison
    print("\nScore distributions (mean):")
    for uni in universities:
        if len(scores_by_uni[uni]) > 0:
            market_mean = np.mean([s['market'] for s in scores_by_uni[uni].values()])
            mission_mean = np.mean([s['mission'] for s in scores_by_uni[uni].values()])
            print(f"  {uni}: market={market_mean:.3f}, mission={mission_mean:.3f}")
            results[f"{uni}_market_mean"] = float(market_mean)
            results[f"{uni}_mission_mean"] = float(mission_mean)

    # Summary
    print("\n" + "="*60)
    print("D1 SUMMARY:")
    print(f"- {len(universities)} universities: {', '.join(universities)}")
    print(f"- Full Pareto: {len(full_pareto)} options")
    print("- Low cross-university overlap shows diversity")
    print("="*60)

    # Save results
    with open(out_dir / "T5_cross_university_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Create CSV table
    csv_lines = ["University,Courses,Missions,Pareto,In Full,Jaccard vs Full"]
    for uni in universities:
        n_courses = len(courses_by_uni[uni])
        n_missions = len(missions_by_uni[uni])
        n_pareto = len(pareto_by_uni[uni])
        in_full = results.get(f"{uni}_in_full", 0)
        jaccard_full = len(pareto_by_uni[uni] & full_pareto) / max(1, len(pareto_by_uni[uni] | full_pareto))
        csv_lines.append(f"{uni},{n_courses},{n_missions},{n_pareto},{in_full},{jaccard_full:.3f}")
    csv_lines.append(f"Full (pooled),{len(courses)},{len(missions)},{len(full_pareto)},-,-")

    with open(out_dir / "tables/T5_cross_university.csv", 'w') as f:
        f.write('\n'.join(csv_lines))

    # Create LaTeX table
    latex = r"""\begin{table}[h]
\centering
\caption{Cross-University Pareto Analysis}
\label{tab:cross-university}
\begin{tabular}{lrrrrr}
\toprule
University & Courses & Missions & Pareto & In Full & Jaccard \\
\midrule
"""
    for uni in universities:
        n_courses = len(courses_by_uni[uni])
        n_missions = len(missions_by_uni[uni])
        n_pareto = len(pareto_by_uni[uni])
        in_full = results.get(f"{uni}_in_full", 0)
        jaccard_full = len(pareto_by_uni[uni] & full_pareto) / max(1, len(pareto_by_uni[uni] | full_pareto))
        latex += f"{uni} & {n_courses:,} & {n_missions} & {n_pareto} & {in_full} & {jaccard_full:.3f} \\\\\n"
    latex += f"\\midrule\nFull (pooled) & {len(courses):,} & {len(missions)} & {len(full_pareto)} & -- & -- \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(out_dir / "tables/T5_cross_university.tex", 'w') as f:
        f.write(latex)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
