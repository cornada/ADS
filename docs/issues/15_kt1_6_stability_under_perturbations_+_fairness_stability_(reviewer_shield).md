# [KT1.6] Stability Under Perturbations + Fairness‑Stability (Reviewer Shield)

## Why
IJCAI reviewers commonly attack:
- “Your embeddings are fragile; results change if the encoder changes.”
- “Pareto sets are unstable; small changes lead to different recommendations.”
- “Fairness/ethics claims are not validated or not robust.”

This milestone creates a **systematic stability evaluation suite**:
1) *Stability of recommendations and Pareto sets* under perturbations.  
2) *Fairness‑stability* (or at minimum, **representation stability**) across runs.  
3) A paper‑ready table/plot that acts as a **reviewer shield**.

This is aligned with ADS contestability: we want controlled, explainable sensitivity, not chaotic behavior.

---

## Scope
### In scope
- Implement perturbation operators:
  - **Embedding noise**: add small Gaussian noise to vectors (post‑embedding) and re‑normalize.
  - **Target bootstrap**: resample stakeholder target sets (courses/roles/missions) and recompute centroids.
  - **Encoder swap**: stub vs SBERT runs (when available).
- Implement stability metrics:
  - **Pareto‑Jaccard**: Jaccard similarity of Pareto option sets between runs.
  - **Rank stability**: Kendall tau / Spearman on aggregated rankings (optional).
  - **Hypervolume stability**: mean/std of hypervolume across runs (optional).
  - **Constraint stability**: percent of options that flip feasible↔infeasible.
- Implement fairness/representation metrics (choose one path depending on available metadata):
  1) If a group field exists (e.g., `course_area`, `department`, `unit`, `topic_cluster`):
     - exposure/coverage per group in top‑K or Pareto set
     - group disparity (max-min or KL divergence) and its stability across runs
  2) If no group field exists:
     - create **pseudo-groups** via k‑means clustering of course embeddings (documented, seeded)
     - treat them as “subarea clusters” and report representation stability (no demographic claims).
- Run the evaluation across datasets:
  - `toy` (required)
  - `mit`, `ucb`, `asu` (once KT1.4 alignment exists; use fixtures if needed)

### Out of scope
- Demographic fairness claims without demographic data
- Causal robustness claims
- Heavy solver-based robustness (RobustX integration is separate)

---

## Deliverables
- [ ] `ads_core/analysis/perturbations.py`
  - `add_embedding_noise(vectors, std, seed)`
  - `bootstrap_targets(items, frac, seed)`
- [ ] `ads_core/analysis/stability.py`
  - `compute_pareto_jaccard(run_a, run_b)`
  - `compute_feasibility_flip_rate(...)`
  - optional: hypervolume + rank stability
- [ ] `ads_core/analysis/representation.py`
  - compute group exposure + disparity
  - compute stability (std/CI) across perturbation runs
- [ ] Hydra config group: `experiments/conf/stability/*.yaml`
  - `enable: true`
  - `n_runs`, `noise_std`, `bootstrap_frac`, `group_field`, `kmeans_k` (if needed)
- [ ] Script: `experiments/scripts/stability_eval.py`
  - runs stability evaluation for selected datasets
  - writes paper artifacts:
    - `experiments/reports/paper/cross_dataset_stability_summary.csv`
    - `experiments/reports/paper/figures/pareto_jaccard_boxplot.png`
    - optional: `feasibility_flip_rate.png`, `group_exposure_stability.png`
- [ ] Tests:
  - deterministic outputs for fixed seed on toy/fixtures
  - pareto_jaccard in [0,1]
  - perturbation operators preserve vector norms (after renorm)

---

## Metrics (minimum)
- **Pareto‑Jaccard**: mean ± std across `n_runs`
- **Feasibility flip rate**: fraction of options whose feasibility changes
- **Representation disparity** (group-based): mean ± std (or CI)

---

## Acceptance criteria (must run and paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `python -m experiments.run dataset=toy embedding=stub lenses=identity`
- [ ] `python experiments/scripts/stability_eval.py --datasets toy --embedding stub --lenses identity --n_runs 10 --noise_std 0.01 --bootstrap_frac 0.8`
- [ ] `python experiments/scripts/stability_eval.py --datasets mit ucb asu --embedding stub --lenses identity --n_runs 5 --noise_std 0.01 --bootstrap_frac 0.8`  
  (If real datasets are unavailable on CI, must run on fixtures and document it.)

Optional (if embed extras installed):
- [ ] `python experiments/scripts/stability_eval.py --datasets toy --embedding sbert_allminilm --lenses learned --n_runs 5`

---

## Expected artifacts
- `experiments/reports/paper/cross_dataset_stability_summary.csv`
- `experiments/reports/paper/figures/pareto_jaccard_boxplot.png`
- (optional) `experiments/reports/paper/figures/group_exposure_stability.png`
- (optional) `experiments/reports/paper/paper_tables/stability_ablation.csv`

---

## Implementation notes
### 1) Keep claims honest
- If using k‑means pseudo‑groups, call it **“subarea representation stability”**, not demographic fairness.
- If group fields exist (department/topic), we can talk about “equitable exposure across subareas”.

### 2) Choose perturbations that match likely reviewer attacks
- Encoder swap and centroid bootstrap are the strongest.
- Noise is cheap and should be included.

### 3) Report both absolute and relative stability
- Absolute: Jaccard mean±std
- Relative: compare stability with identity vs learned lenses (does lens learning destabilize?)

---

## Prompt for Cursor Composer
> Implement KT1.6: add perturbation operators (noise + bootstrap targets), stability metrics (Pareto‑Jaccard, feasibility flip), and representation stability metrics (group field or k‑means pseudo-groups). Add hydra configs and a stability_eval.py script that outputs cross_dataset_stability_summary.csv and pareto_jaccard_boxplot.png. Add tests and fixtures. Keep toy run green.

## Prompt for Claude Code
> Read docs/issues/15_kt1_6_stability_under_perturbations_+_fairness_stability_(...).md. Propose a plan (files/tests/commands). Implement perturbations, stability metrics, and the stability_eval script. Run pytest and the stability_eval command for toy, and report output paths to paper artifacts.
