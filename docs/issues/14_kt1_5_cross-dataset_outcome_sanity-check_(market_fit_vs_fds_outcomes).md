# [KT1.5] Cross‑Dataset Outcome Sanity‑Check (MIT + UC Berkeley + ASU): market‑fit vs outcomes

## Why
ADS must show at least one **empirical grounding** beyond “nice visualizations”.
A lightweight but reviewer‑credible way (no PII) is:

- Compute **market‑fit** of a program/curriculum representation against market targets (O*NET/ESCO roles/skills).
- Compare the predicted market‑fit distributions to **aggregated outcomes** (e.g., Berkeley First Destination Survey categories by major/college; ASU career outcomes; MIT post‑grad outcomes where available).
- Benchmark against a simple baseline (e.g., “major name only” classifier / keyword match).

This is explicitly aligned with the ADS strategy: open data, no LinkedIn as primary source, reproducible outcome validation.

---

## Scope
### In scope
- Define a **canonical outcomes schema** (if not already done in KT1.4):
  - keys: `institution`, `unit` (college/major/program), `cohort_year`, `category`, `value`, `source_url`, `raw_hash`
- Define **canonical market categories**:
  - either from O*NET/ESCO role taxonomy (recommended), or
  - a small manually curated mapping for MVP (documented, versioned).
- Implement an evaluator:
  1) represent each `unit` (major/program/college) by:
     - centroid of its courses (catalog/OCW where available), OR
     - mission + course centroid (two variants for ablation)
  2) compute similarity to market targets (roles/categories)
  3) compare to observed outcome distribution
- Produce paper‑ready outputs:
  - `cross_dataset_outcome_sanity.csv` (main numbers)
  - `outcome_sanity_scatter.png` (fit vs observed)
  - optional: `ablation_table.csv` (course-only vs mission+course)

### Out of scope
- Individual‑level prediction (no PII)
- Causal claims (“this causes employment”)
- Full KT/CD student modeling

---

## Deliverables
- [ ] `docs/datasets/outcomes_schema.md` (or add section to `unified_schema.md`)
- [ ] `ads_core/outcomes/normalize.py`
  - normalize UCB FDS exports and ASU reports into canonical outcomes table
  - store provenance fields (`source_url`, `retrieved_at`, `raw_hash`, `license`)
- [ ] `ads_core/market/targets.py`
  - create market target vectors from O*NET/ESCO roles/skills (or a documented minimal list)
- [ ] `ads_core/analysis/outcome_sanity.py`
  - compute:
    - predicted distribution over categories (via similarity),
    - correlation / rank correlation to observed outcomes,
    - calibration error (optional)
- [ ] `experiments/scripts/outcome_sanity.py`
  - runs the pipeline for datasets `mit`, `ucb`, `asu`
  - writes `experiments/reports/paper/cross_dataset_outcome_sanity.csv`
  - writes `experiments/reports/paper/figures/outcome_sanity_scatter.png`
- [ ] Baselines:
  - B0: “major name only” keyword baseline (documented)
  - B1: content centroid baseline (course text centroid)
- [ ] Tests:
  - normalization creates required columns
  - deterministic outputs on fixtures

---

## Metrics (minimum)
- Spearman correlation between predicted category ranking and observed outcome share (per unit)
- Mean Absolute Error (MAE) between predicted and observed distributions (per unit)
- Coverage: fraction of units with usable outcomes data

---

## Acceptance criteria (must run and paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `python -m experiments.run dataset=toy embedding=stub lenses=identity`
- [ ] `python experiments/scripts/outcome_sanity.py --datasets ucb asu --embedding stub --lenses identity`
- [ ] `python experiments/scripts/outcome_sanity.py --datasets ucb asu --embedding sbert_allminilm --lenses identity` (optional if embed extras installed)

> MIT outcomes are optional depending on availability of open outcome aggregates.
> If MIT outcomes aren’t reliably extractable, implement UCB+ASU first and leave MIT as TODO with a documented plan.

---

## Expected artifacts
- `experiments/reports/paper/cross_dataset_outcome_sanity.csv`
- `experiments/reports/paper/figures/outcome_sanity_scatter.png`
- (optional) `experiments/reports/paper/paper_tables/ablation_outcome_sanity.csv`

---

## Implementation notes (recommended approach)
### 1) Outcomes extraction strategy
- UC Berkeley: allow **manual CSV export** from dashboards if direct download is not available.
  - Store exported file(s) in `data/raw/ucb/fds/`
  - Record a `manifest.yaml` describing how export was produced + date/time.
- ASU: parse PDF/HTML outcomes reports into canonical table.
- MIT: if an open outcomes PDF/dashboard exists, treat it similarly; otherwise mark as TODO.

### 2) Category alignment (keep it honest)
- Do NOT overfit mappings.
- Start with 6–12 broad categories (e.g., software/ML, data, research, policy, finance, healthcare, etc.).
- Document mapping rules and include them in repo version control.

### 3) Avoid leakage
- Do not use outcomes text as part of the “course representation” inputs.

---

## Prompt for Cursor Composer
> Implement KT1.5: outcome sanity-check pipeline. Normalize UCB+ASU outcomes into canonical table with provenance. Build market targets from O*NET/ESCO (or a minimal documented list). Compute similarity-based predicted distributions and compare to observed outcomes (Spearman + MAE). Output cross_dataset_outcome_sanity.csv and outcome_sanity_scatter.png. Add tests and fixtures. Keep toy run green.

## Prompt for Claude Code
> Read docs/issues/14_kt1_5_cross-dataset_outcome_sanity-check_(...).md. Propose a plan (files/tests/commands). Implement outcomes normalization + market targets + analysis script + report artifacts. Run pytest and the outcome_sanity script on fixtures and report output paths.
