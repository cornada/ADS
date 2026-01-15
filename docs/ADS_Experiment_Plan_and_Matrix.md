# ADS Experiment Plan & Matrix (handoff-ready)
**Project:** Agent‑Didactic Spaces (ADS)  
**Audience:** colleague running experiments + preparing figures/tables for IJCAI  
**Date:** 2026‑01‑14

---

## 1) Research questions (RQ) — what we must answer to be “unbreakable”

### RQ1 — Do multi-objective (Pareto) options provide better decision support than scalar top‑k?
- Hypothesis: Pareto dashboard increases **diversity of chosen pathways** and **perceived autonomy**, without reducing market‑fit.

### RQ2 — Do stakeholder lenses materially improve evaluation quality vs “one embedding space without lenses”?
- Hypothesis: lenses improve interpretability and alignments to external anchors (market/mission).

### RQ3 — Does ethics/fairness constraint prevent “capture” without collapsing utility?
- Hypothesis: constraints reduce autonomy‑drift and improve fairness stability, with small cost in market objective.

### RQ4 — Are explanations contestable and actionable?
- Hypothesis: counterfactual/recourse suggestions are small (low cost), robust, and lead to feasible alternatives on Pareto frontier.

---

## 2) Datasets (prioritize open + reproducible)

### D1 — Open Curriculum Corpus (no students)
**Purpose:** feasibility + Pareto front exists and interpretable.  
**Sources (examples):**
- Program/department mission statements
- Course catalog descriptions + (if available) syllabi/OCW materials
- Market anchors: O*NET/ESCO role + skill descriptions
- Research anchors: OpenAlex topics/abstracts (optional)

**Artifacts:**
- Courses: 30–80
- Job roles: 10–30
- Mission texts: 1–3

### D2 — Outcomes aggregates (no PII)
**Purpose:** sanity-check that content-based fit predicts outcomes better than “major name”.  
**Sources:**
- public first-destination dashboards / aggregated employment categories (as available)

### D3 — KT benchmark dataset (optional but strong)
**Purpose:** include “learning gain / mastery” objective with established baselines.  
**Candidates:**
- ASSISTments / Junyi / other standard KT datasets

### D4 — Small human study dataset (minimal n=10–20)
**Purpose:** show autonomy/contestability benefit.  
**Data:** short learner self‑statement + preferences, chosen options; no PII beyond consent.

---

## 3) Methods under test (ADS variants)

### ADS‑Base
- Single encoder embeddings
- Lenses = identity (no lens)
- Objectives: market/mission/univ/learner distances
- Output: scalar top‑k (weighted sum)

### ADS‑Pareto (core)
- Same as base but output Pareto set (nondominated)

### ADS‑Lenses
- Lenses enabled (diagonal or learned)

### ADS‑Ethics
- Ethics constraint enabled (autonomy drift / fairness constraint)

### ADS‑Recourse
- Adds counterfactual/recourse suggestions (heuristic or RobustX)

---

## 4) Baselines (must include to satisfy reviewers)

### B0 — Similarity ranking (very basic)
- cosine similarity to one target (market) OR weighted sum

### B1 — Multi-objective bandit (MONB) for “next step”
- random scalarization; compare to ADS sequential selection

### B2 — Fair rank aggregation
- convert multi-objective scores into a single fair ranking

### B3 — KT / student modeling baselines (if D3 is included)
- pyKT/EduStudio standard models; (optional) TransKT if cross-course concept graph is used

---

## 5) Objective definitions (what ADS optimizes)

Let `v(s)` be learner/profile state vector, `v(a)` option vector, `T_market`, `T_mission`, `T_univ`.

### Core objectives (distance-based; lower=better)
- `d_market(a)` = dist( L_market(v(a)), centroid(L_market(T_market)) )
- `d_mission(a)` similarly
- `d_univ(a)` similarly
- `d_learner(a)` = dist( L_learner(v(a)), v(s) ) or dist to learner goal statement

### Learning objective (if KT)
- `gain(a|s)` = predicted mastery improvement or success probability

### Cost/time objective (proxy)
- `cost(a)` = credits / estimated hours / prereq depth (simple proxy)

### Ethics/Autonomy objective
- `autonomy_drift(path)` = trajectory functional (see implementation spec)

### Fairness objectives (group-based; if D4 or demographic proxies available)
- equality of opportunity proxy; demographic parity of recommendations; stability metrics

---

## 6) Metrics (what we report)

### 6.1 Pareto quality (offline)
- Pareto set size / coverage
- Hypervolume (normalized)
- ε-indicator
- Stability across seeds / embedding models (Jaccard of Pareto sets)

### 6.2 Utility / alignment
- Market-fit correlation with external outcomes (D2)
- Mission alignment score distribution
- Average distance improvements vs baseline

### 6.3 Diversity
- Diversity of selected options (entropy over categories, average pairwise distance)
- Coverage across clusters/topics

### 6.4 Ethics/fairness
- Autonomy drift distribution
- Fairness gaps (as applicable)
- Fairness stability under perturbations (from FAiRDAS style)

### 6.5 Explanation/recourse
- Recourse cost: number of changes, distance moved, added steps
- Robustness: sensitivity to small perturbations
- Actionability: percent of recourses that lead to feasible Pareto alternatives

### 6.6 Human study (D4)
- Perceived autonomy (Likert)
- Satisfaction / trust
- Time-to-decision
- Chosen option diversity
- Preference consistency (post-hoc)

---

## 7) Experiment matrix (dataset × method × baseline × ablations)

### Table A — Core feasibility (D1)
| Exp | Dataset | Methods | Baselines | Ablations | Outputs |
|---|---|---|---|---|---|
| A1 | D1 | ADS‑Pareto | B0 | encoder A vs encoder B | Pareto plots + stability |
| A2 | D1 | ADS‑Lenses | ADS‑Pareto(no lens) | diagonal vs learned lens | interpretability + metrics |
| A3 | D1 | ADS‑Ethics | ADS‑Lenses(no ethics) | τ sweep | autonomy drift vs utility |

### Table B — Outcomes sanity check (D2)
| Exp | Dataset | Methods | Baselines | Ablations | Outputs |
|---|---|---|---|---|---|
| B1 | D2 | ADS‑Lenses | major-name baseline | target definitions | correlation/fit plots |

### Table C — KT objective integration (D3, optional)
| Exp | Dataset | Methods | Baselines | Ablations | Outputs |
|---|---|---|---|---|---|
| C1 | D3 | ADS + KT objective | KT-only | with/without KT objective | tradeoff curves |
| C2 | D3 | ADS + TransKT (if used) | pyKT models | graph construction variants | mastery/fit tradeoffs |

### Table D — Human study (D4)
| Exp | Dataset | Methods | Baselines | Ablations | Outputs |
|---|---|---|---|---|---|
| D1 | D4 | ADS‑Pareto+Explain | scalar top‑k | with/without recourse | autonomy/satisfaction/diversity |

---

## 8) Reporting template (paper-ready)

### Figures (recommended)
1) 2D Pareto scatter (market vs mission) with nondominated highlighted
2) Radar charts for 3–5 representative Pareto options
3) Stability plot (Pareto Jaccard across encoders/seeds)
4) Ethics tradeoff curve (autonomy drift vs market fit)
5) Recourse example: before/after on Pareto plane
6) (If D4) user study barplots for autonomy/satisfaction

### Tables (recommended)
- Table: Dataset stats (counts, sources)
- Table: Main results (Pareto metrics + alignment)
- Table: Ablations (lens/ethics/encoder)
- Table: Human study summary (means, effect sizes)

---

## 9) Minimal reproducibility protocol (must follow)
- Each experiment run writes:
  - `config.yaml`, `git_commit`, `random_seed`, `model_id`
  - dataset hashes and provenance logs
- Save outputs to `experiments/reports/<exp_id>/`
- Provide `make reproduce_all_small` that regenerates key figures on toy data.

---

## 10) What to implement first (priority order)
1) D1 + A1/A2 (fastest “IJCAI-looking” results)
2) A3 (ethics constraint curve)
3) D2 (outcomes sanity check)
4) D4 (small human study)
5) D3 (KT integration) if you want stronger “learning” objective beyond text similarity
