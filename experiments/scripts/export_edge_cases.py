"""Export edge cases with course details for paper appendix."""
import argparse
import json
import numpy as np
from pathlib import Path
from hashlib import sha256

_REPO_ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description="Export edge cases for paper appendix")
    parser.add_argument('--run-dir', type=str,
                        default=str(_REPO_ROOT / 'experiments/reports/latest'))
    parser.add_argument('--output-dir', type=str,
                        default=str(_REPO_ROOT / 'experiments/reports/edge_cases'))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    cache_dir = run_dir / "cache_embeddings" / "sbert_sentence-transformers_all-MiniLM-L6-v2"
    embeddings = np.load(cache_dir / "vecs.npy")
    with open(cache_dir / "index.json", encoding='utf-8') as f:
        index = json.load(f)

    with open(run_dir / "store" / "artifacts.jsonl", encoding='utf-8') as f:
        artifacts = [json.loads(line) for line in f]

    text_to_idx = {}
    artifact_by_id = {}
    for a in artifacts:
        h = sha256(a['text'].encode()).hexdigest()
        if h in index:
            text_to_idx[a['artifact_id']] = index[h]
        artifact_by_id[a['artifact_id']] = a

    courses = [a for a in artifacts if a['type'] == 'COURSE']
    jobs = [a for a in artifacts if a['type'] == 'JOB_ROLE']
    missions = [a for a in artifacts if a['type'] == 'MISSION']

    # Compute centroids
    job_embs = np.array([embeddings[text_to_idx[j['artifact_id']]]
                         for j in jobs if j['artifact_id'] in text_to_idx])
    mission_embs = np.array([embeddings[text_to_idx[m['artifact_id']]]
                             for m in missions if m['artifact_id'] in text_to_idx])

    market_c = np.mean(job_embs, axis=0)
    market_c = market_c / np.linalg.norm(market_c)
    mission_c = np.mean(mission_embs, axis=0)
    mission_c = mission_c / np.linalg.norm(mission_c)

    # Compute scores
    course_scores = []
    for c in courses:
        if c['artifact_id'] in text_to_idx:
            emb = embeddings[text_to_idx[c['artifact_id']]]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            text = c['text']
            # Extract title (first line or first 100 chars)
            title = text.split('\n')[0][:100] if '\n' in text else text[:100]
            course_scores.append({
                'id': c['artifact_id'],
                'market': float(np.dot(emb_norm, market_c)),
                'mission': float(np.dot(emb_norm, mission_c)),
                'university': 'UC Berkeley' if ':UC Berkeley:' in c['artifact_id'] else 'MIT',
                'title': title,
                'text_preview': text[:300] + '...' if len(text) > 300 else text
            })

    # Compute Pareto
    pareto_courses = []
    for i, c_i in enumerate(course_scores):
        is_pareto = True
        for j, c_j in enumerate(course_scores):
            if i != j:
                if c_j['market'] >= c_i['market'] and c_j['mission'] >= c_i['mission']:
                    if c_j['market'] > c_i['market'] or c_j['mission'] > c_i['mission']:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_courses.append(c_i)

    print(f"Found {len(pareto_courses)} Pareto courses")

    # Find edge cases
    max_market = max(pareto_courses, key=lambda x: x['market'])
    max_mission = max(pareto_courses, key=lambda x: x['mission'])
    balanced = min(pareto_courses, key=lambda x: abs(x['market'] - x['mission']))

    # Find closest job for max_market
    max_market_emb = embeddings[text_to_idx[max_market['id']]]
    max_market_emb = max_market_emb / np.linalg.norm(max_market_emb)
    best_job_sim = -1
    best_job = None
    for j in jobs:
        if j['artifact_id'] in text_to_idx:
            job_emb = embeddings[text_to_idx[j['artifact_id']]]
            job_emb = job_emb / np.linalg.norm(job_emb)
            sim = float(np.dot(max_market_emb, job_emb))
            if sim > best_job_sim:
                best_job_sim = sim
                best_job = j

    # Generate markdown
    md_content = f"""# Edge Case Analysis - Paper Appendix

Generated: 2026-01-17

## Summary

| Edge Case | Course ID | University | Market | Mission |
|-----------|-----------|------------|--------|---------|
| Max Market | {max_market['id'][:30]}... | {max_market['university']} | {max_market['market']:.3f} | {max_market['mission']:.3f} |
| Max Mission | {max_mission['id'][:30]}... | {max_mission['university']} | {max_mission['market']:.3f} | {max_mission['mission']:.3f} |
| Most Balanced | {balanced['id'][:30]}... | {balanced['university']} | {balanced['market']:.3f} | {balanced['mission']:.3f} |

---

## 1. Maximum Market Alignment

**Course ID**: `{max_market['id']}`

**University**: {max_market['university']}

**Scores**:
- Market: {max_market['market']:.4f}
- Mission: {max_market['mission']:.4f}

**Title/Preview**:
> {max_market['title']}

**Text Preview**:
```
{max_market['text_preview']}
```

**Closest Job Role**: `{best_job['artifact_id'] if best_job else 'N/A'}`
- Similarity: {best_job_sim:.4f}
- Preview: {best_job['text'][:200] if best_job else 'N/A'}...

---

## 2. Maximum Mission Alignment

**Course ID**: `{max_mission['id']}`

**University**: {max_mission['university']}

**Scores**:
- Market: {max_mission['market']:.4f}
- Mission: {max_mission['mission']:.4f}

**Title/Preview**:
> {max_mission['title']}

**Text Preview**:
```
{max_mission['text_preview']}
```

---

## 3. Most Balanced (Market ~ Mission)

**Course ID**: `{balanced['id']}`

**University**: {balanced['university']}

**Scores**:
- Market: {balanced['market']:.4f}
- Mission: {balanced['mission']:.4f}
- Difference: {abs(balanced['market'] - balanced['mission']):.4f}

**Title/Preview**:
> {balanced['title']}

**Text Preview**:
```
{balanced['text_preview']}
```

---

## Interpretation

1. **Max Market** course is strongly aligned with job market demands, likely a technical/vocational course with direct industry applications.

2. **Max Mission** course embodies university educational mission - likely interdisciplinary, research-focused, or addressing societal challenges.

3. **Balanced** course represents a trade-off point where both stakeholder objectives are reasonably satisfied.

The existence of these distinct edge cases demonstrates that the Pareto frontier captures genuine trade-offs between market demand and institutional mission.
"""

    with open(output_dir / 'edge_cases.md', 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"Saved: {output_dir / 'edge_cases.md'}")

    # Also save as JSON
    edge_cases_json = {
        'max_market': max_market,
        'max_mission': max_mission,
        'balanced': balanced,
        'closest_job_to_max_market': {
            'id': best_job['artifact_id'] if best_job else None,
            'similarity': best_job_sim,
            'preview': best_job['text'][:200] if best_job else None
        }
    }

    with open(output_dir / 'edge_cases.json', 'w', encoding='utf-8') as f:
        json.dump(edge_cases_json, f, indent=2)

    print(f"Saved: {output_dir / 'edge_cases.json'}")

if __name__ == "__main__":
    main()
