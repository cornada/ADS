# ADS Benchmark: Task-Agnostic Evaluation Protocol

**Version:** 1.0
**Date:** 2026-02-07
**Status:** Designed for KDD 2026 Datasets & Benchmarks Track

## 1. Motivation

Reviewers identified that the original benchmark task ("compute a feasible Pareto set under dominance constraint tau using ADS lenses") is tightly coupled to the ADS formalism, preventing external teams from bringing alternative methods. This protocol defines three benchmark tasks purely in terms of observable inputs and outputs, so that **any method** -- ADS, TF-IDF + weighted sum, LLM-based rankers, NSGA-II, or hand-curated lists -- can participate without importing ADS code.

## 2. Benchmark Tasks

### Task A: Multi-Stakeholder Course Ranking

**Goal:** Given course text artifacts and stakeholder target corpora, produce a ranked set of courses that balances multiple stakeholder objectives.

**Input:**
- Course text corpus C (course descriptions, titles, learning outcomes)
- Stakeholder target corpora {T_k} for k in {market, mission, university, learner}
  - market: O*NET occupation descriptions
  - mission: Institutional mission statements
  - university: Existing curriculum descriptions
  - learner: Learner goal descriptions

**Output:** Ordered list of course IDs with optional per-course score vectors.

**Metrics:**
| Metric | Definition | Direction |
|--------|-----------|-----------|
| Coverage | Fraction of objective-space grid cells (5-bin discretization) occupied by selected set | Higher is better |
| Diversity | Mean pairwise Euclidean distance in ground-truth objective space | Higher is better |
| Objective Balance | Normalized Shannon entropy of mean objective proportions | Higher is better (max = 1.0) |
| External Alignment (Precision) | Fraction of selected courses on reference Pareto front | Higher is better |
| External Alignment (Recall) | Fraction of reference Pareto front captured | Higher is better |
| External Alignment (F1) | Harmonic mean of precision and recall | Higher is better |
| External Alignment (Jaccard) | Set overlap with reference Pareto front | Higher is better |
| Rank Quality | Spearman correlation with aggregate-score ranking | Higher is better |

### Task B: Stakeholder Trade-off Discovery

**Goal:** Identify a set of courses that represent distinct trade-offs across stakeholder objectives (i.e., approximate the Pareto frontier).

**Input:** Same as Task A.

**Output:** Set of course IDs representing distinct trade-off points, with optional score vectors.

**Metrics:**
| Metric | Definition | Direction |
|--------|-----------|-----------|
| Hypervolume (HV) | Volume of objective space dominated by submitted set (ref=**0**) | Higher is better |
| Spacing | Std dev of nearest-neighbor distances in objective space | Lower is better |
| Spread | Bounding-box volume of submitted set in objective space | Higher is better |
| Pareto Precision | Fraction of submitted courses on true Pareto front | Higher is better |
| Pareto Recall | Fraction of true Pareto front included in submission | Higher is better |
| Dominance Resistance | Fraction of submitted points non-dominated within submission | Higher is better |
| Objective Balance | Normalized Shannon entropy of mean objective proportions | Higher is better |

### Task C: Robust Recommendation

**Goal:** Produce recommendations that are stable under encoder perturbation and stakeholder weight variation.

**Input:** Same as Task A, plus a perturbation protocol:
- Encoder perturbation: Gaussian noise sigma=0.01 added to embeddings
- Weight perturbation: Gaussian noise sigma=0.1 added to objective weights

**Output:** Primary submission + at least 2 perturbation-run submissions.

**Metrics:**
| Metric | Definition | Direction |
|--------|-----------|-----------|
| Bootstrap Jaccard (mean) | Mean pairwise Jaccard of course sets across runs | Higher is better |
| Bootstrap Jaccard (std) | Std dev of pairwise Jaccard | Lower is better |
| HV Retention | min(HV) / max(HV) across perturbation runs | Higher is better (max=1.0) |
| Rank Stability (mean) | Mean pairwise Spearman correlation of rankings | Higher is better |
| Core Stable Fraction | Fraction of courses appearing in >=80% of runs | Higher is better |

## 3. Submission Format (JSON Schema)

```json
{
  "method_name": "string (required)",
  "task": "A | B | C | all (required)",
  "ranked_courses": ["course_id_1", "course_id_2", "..."],
  "scores": {
    "course_id_1": {
      "market": 0.82,
      "mission": 0.31,
      "university": 0.45,
      "learner": 0.50
    }
  },
  "perturbation_runs": [
    {
      "ranked_courses": ["course_id_1", "course_id_2", "..."],
      "scores": {}
    }
  ]
}
```

**Field descriptions:**
- `method_name` (string, required): Name of the submitted method.
- `task` (string, required): Which task(s) to evaluate. One of "A", "B", "C", "all".
- `ranked_courses` (list of strings, required): Course IDs in ranked order (best first). For Task A, this is the ranking. For Task B, order does not matter but must be present.
- `scores` (dict, optional): Per-course score vectors. If provided, objective keys must match the ground-truth objectives. If not provided, ground-truth scores are used for evaluation.
- `perturbation_runs` (list, required for Task C): Each element has `ranked_courses` and optional `scores`. At least 2 runs required.

## 4. Ground-Truth Format (JSON Schema)

```json
{
  "objectives": ["market", "mission", "university", "learner"],
  "courses": {
    "course_id": {
      "market": 0.82,
      "mission": 0.31,
      "university": 0.45,
      "learner": 0.50
    }
  },
  "reference_pareto": ["course_id_1", "course_id_7", "..."],
  "splits": {
    "dev": ["course_id_1", "course_id_2"],
    "eval": ["course_id_1", "course_id_2", "course_id_3"]
  }
}
```

**Field descriptions:**
- `objectives` (list of strings, required): Names of the stakeholder objectives.
- `courses` (dict, required): All courses with their ground-truth objective scores.
- `reference_pareto` (list, optional): Course IDs on the true Pareto front (computed from ground-truth scores if not provided).
- `splits` (dict, optional): Development and evaluation course ID sets.

## 5. Evaluation Script

**Location:** `experiments/scripts/benchmark_eval.py`

**Dependencies:** Python 3.8+, numpy, scipy (optional, for Spearman correlation; falls back to manual computation).

**Usage:**

```bash
# Evaluate a submission
python benchmark_eval.py --submission sub.json --ground-truth gt.json --output results.json --pretty

# Evaluate specific task only
python benchmark_eval.py --submission sub.json --ground-truth gt.json --task B

# Run self-test with synthetic data
python benchmark_eval.py --self-test

# Generate example files for development
python benchmark_eval.py --generate-example --output examples/
```

The script is fully self-contained: it re-implements Pareto dominance, hypervolume, spacing, and all metrics without importing any ADS package code.

## 6. How External Methods Participate

### Example 1: TF-IDF + Weighted Sum

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# 1. Load course texts and stakeholder corpora
courses = load_course_texts()        # {course_id: text}
market_corpus = load_onet_texts()    # [text1, text2, ...]
mission_corpus = load_missions()     # [text1, text2, ...]

# 2. Build TF-IDF representations
vectorizer = TfidfVectorizer()
all_texts = list(courses.values()) + market_corpus + mission_corpus
tfidf = vectorizer.fit_transform(all_texts)

# 3. Score each course against each stakeholder centroid
course_vecs = tfidf[:len(courses)]
market_centroid = tfidf[len(courses):len(courses)+len(market_corpus)].mean(axis=0)
mission_centroid = tfidf[-len(mission_corpus):].mean(axis=0)

market_scores = cosine_similarity(course_vecs, market_centroid).flatten()
mission_scores = cosine_similarity(course_vecs, mission_centroid).flatten()

# 4. Rank by weighted sum and format submission
ids = list(courses.keys())
weights = [0.5, 0.5]
total = [w_m * m + w_n * n for w_m, m, w_n, n in zip(
    [weights[0]]*len(ids), market_scores,
    [weights[1]]*len(ids), mission_scores)]
ranked = sorted(range(len(ids)), key=lambda i: -total[i])

submission = {
    "method_name": "tfidf_weighted_sum",
    "task": "B",
    "ranked_courses": [ids[i] for i in ranked[:100]],
    "scores": {ids[i]: {"market": float(market_scores[i]),
                         "mission": float(mission_scores[i])}
               for i in ranked[:100]}
}
with open("submission.json", "w") as f:
    json.dump(submission, f)
```

### Example 2: LLM-Based Ranker

```python
import json
from openai import OpenAI

client = OpenAI()
courses = load_course_texts()

# Ask LLM to rank courses for each stakeholder
def llm_score(course_text, stakeholder_desc):
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user",
                   "content": f"Rate 0-1 how well this course aligns with "
                              f"this stakeholder goal.\nCourse: {course_text}\n"
                              f"Goal: {stakeholder_desc}\nScore:"}]
    )
    return float(resp.choices[0].message.content.strip())

# Score all courses (simplified)
scores = {}
for cid, text in courses.items():
    scores[cid] = {
        "market": llm_score(text, "labor market employability"),
        "mission": llm_score(text, "university educational mission"),
    }

ranked = sorted(scores, key=lambda c: sum(scores[c].values()), reverse=True)
submission = {
    "method_name": "gpt4_ranker",
    "task": "A",
    "ranked_courses": ranked[:100],
    "scores": {c: scores[c] for c in ranked[:100]}
}
```

## 7. Design Principles

1. **Formalism-agnostic:** Tasks are defined by inputs (texts) and outputs (ranked IDs), not by internal representations or algorithms.

2. **Metric separation:** Metrics evaluate the *quality of the output*, not the method's internal structure. Any method producing the same output gets the same score.

3. **Self-contained evaluation:** The evaluation script re-implements all metrics from scratch using only numpy/scipy. No ADS imports needed.

4. **Multiple evaluation dimensions:** Rather than a single leaderboard score, we report per-task metric profiles. A method may excel at Task A (ranking) while being weak at Task C (robustness).

5. **Ground-truth from data, not from ADS:** Ground-truth scores are computed from course-corpus similarity (which any embedding method can reproduce). The reference Pareto front is derived from these scores, not from ADS-specific computations.

## 8. Splits and Evaluation Protocol

- **Development split:** 3 US universities (~10,400 courses). Use for method development and hyperparameter tuning.
- **Evaluation split:** All universities, all countries. Final benchmark numbers reported on this split.
- **Seeds:** Report results averaged over seeds {0, 1, 2} where applicable (esp. Task C perturbation runs).

## 9. Changelog

- **v1.0 (2026-02-07):** Initial task-agnostic protocol with Tasks A, B, C. Self-contained evaluation script. JSON schemas defined.
