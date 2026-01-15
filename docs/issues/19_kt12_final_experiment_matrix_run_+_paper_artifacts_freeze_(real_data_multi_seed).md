# [KT12] Final experiment matrix run + paper artifacts freeze (real data, multi-seed)

**Generated:** 2026-01-15  
**Owner:** Engineering + Paper lead

## Goal
Run the **final experiment matrix** on the real (non-fixture) datasets and freeze:
- tables (CSV) and figures (PNG/PDF) used in the paper,
- a manifest mapping every paper table/figure to run_id + data_hash + code_hash.

This KT is about making experiments **auditably reproducible**, not just "it works on my machine".

## Why (reviewer risk addressed)
Reviewers will attack:
- cherry-picked runs,
- missing ablations,
- unclear baselines,
- non-reproducible figures.

This KT makes the results traceable and stable.

## Inputs
- Datasets built in KT1.1–KT1.3 (MIT/UCB/ASU) + market_v1
- Alignment and unified schema from KT1.4
- Outcome sanity-check from KT1.5
- Stability suite from KT1.6
- Repro bundle from KT9

## Deliverables
### A) Experiment matrix spec (frozen)
- `experiments/EXPERIMENT_MATRIX_FINAL.yaml`
Includes:
- datasets: mit_v1, ucb_v1, asu_v1
- embeddings: at least (stub, sbert_allminilm OR bge_small)
- lenses: identity + diagonal (and learned if stable)
- objectives: market/mission/university/learner + constraints
- seeds: at least 3 seeds for key results

### B) Frozen paper artifacts directory (stable names)
- `experiments/reports/paper/`
Required outputs (minimum):
- `paper_tables/cross_dataset_summary.csv`
- `paper_tables/cross_dataset_outcome_sanity.csv`
- `paper_tables/stability_summary.csv`
- `paper_figures/pareto_frontier_example.png`
- `paper_figures/outcome_sanity_scatter.png`
- `paper_figures/stability_under_perturbations.png`

### C) Paper results manifest (traceability)
- `experiments/reports/paper/PAPER_RESULTS_MANIFEST.json`
For each table/figure:
- artifact_name
- file_path
- script_path
- run_ids (list)
- data_hash
- code_hash
- generated_at

### D) Deterministic re-run command (small)
- `docs/paper/REPRODUCE_KEY_RESULTS.md`
Must include:
- minimal commands to reproduce key figures/tables (or subset if heavy)
- environment notes (GPU optional)

## Acceptance criteria
- [ ] `pytest -q`
- [ ] `python experiments/scripts/compare_datasets.py ...` regenerates cross_dataset_summary.csv
- [ ] `python experiments/scripts/outcome_sanity.py ...` regenerates cross_dataset_outcome_sanity.csv
- [ ] Stability summary exists and uses >= 3 seeds for main results
- [ ] PAPER_RESULTS_MANIFEST.json has entries for every table/figure referenced in OUTLINE matrix (KT10)

## Prompt for Claude Code (copy/paste)
Create experiments/EXPERIMENT_MATRIX_FINAL.yaml, and implement (or update) runners so that:
- they write paper artifacts into experiments/reports/paper/ with stable filenames,
- they also write PAPER_RESULTS_MANIFEST.json linking artifacts to run_ids/data_hash/code_hash.

Run the minimal regeneration commands and confirm outputs exist.
In PR description paste:
- list of produced files under experiments/reports/paper/
- 5 lines from PAPER_RESULTS_MANIFEST.json (keys only)
