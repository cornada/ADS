# [KT1.4] Cross‑University Alignment (MIT + UC Berkeley + ASU): unified schema + dataset registry + common evaluator

## Why
After KT1.1–KT1.3 we will have **three separate ingestion pipelines** and datasets.  
To make ADS **paper‑grade and reproducible**, we must:
- normalize everything into **one canonical Artifact schema**,
- run the **same evaluator** across institutions (same embeddings / lenses / objectives),
- produce **comparable tables/figures** and a single reproduce script.

This milestone is the “glue” that turns 3 datasets into a coherent IJCAI‑style experimental section.

---

## Scope
### In scope
- A **unified schema** for:
  - missions/program pages,
  - course catalog entries (+ optional OCW materials),
  - outcomes / FDS aggregates,
  - market anchors (O*NET/ESCO roles+skills) (if included),
  - research anchors (OpenAlex) (optional).
- A **dataset registry** with dataset IDs: `toy`, `mit`, `ucb`, `asu`.
- A **common evaluator interface** that runs:
  - embeddings,
  - lenses,
  - objective computation,
  - Pareto extraction,
  - constraint checks,
  - report generation
  with identical outputs for all datasets.
- A **cross‑dataset reporting script** that outputs a comparative table:
  - dataset stats,
  - Pareto set size,
  - stability (optional),
  - a small set of exemplar options.

### Out of scope
- Full UI integration
- KT/CD student modeling (separate milestone)
- “Perfect” ingestion for every department (we can start with 1 dept per uni)

---

## Deliverables
- [ ] `docs/datasets/unified_schema.md` describing:
  - entity types,
  - required fields,
  - provenance/license fields,
  - how outcomes are represented and compared.
- [ ] `ads_core/datasets/registry.py` (or similar) with:
  - `load_dataset(dataset_id, data_dir) -> DatasetBundle`
  - `DatasetBundle.artifacts` + optional `outcomes` tables
- [ ] Canonical processed outputs:
  - `data/processed/mit/artifacts.jsonl`
  - `data/processed/ucb/artifacts.jsonl`
  - `data/processed/asu/artifacts.jsonl`
  - **All records** contain provenance: `source_url`, `retrieved_at`, `raw_hash`, `license`, `institution`.
- [ ] `experiments/conf/dataset/mit.yaml`, `ucb.yaml`, `asu.yaml` with paths to the processed artifacts
- [ ] `experiments/scripts/compare_datasets.py` (or `experiments/compare.py`) that:
  - runs evaluator for each dataset with the same config,
  - writes `experiments/reports/paper/cross_dataset_summary.csv`
- [ ] Tests:
  - dataset loader returns non‑empty artifacts for fixtures,
  - schema validation (required fields present),
  - `dataset_id` registry works.

---

## Acceptance criteria (must run and paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `python -m experiments.run dataset=toy embedding=stub lenses=identity`
- [ ] `python -m experiments.run dataset=mit embedding=stub lenses=identity`
- [ ] `python -m experiments.run dataset=ucb embedding=stub lenses=identity`
- [ ] `python -m experiments.run dataset=asu embedding=stub lenses=identity`
- [ ] `python experiments/scripts/compare_datasets.py --datasets mit ucb asu --embedding stub --lenses identity`

> If real data isn’t available on the CI machine, include **fixtures**:
> `data/fixtures/mit/`, `data/fixtures/ucb/`, `data/fixtures/asu/` and configure registry to fall back to fixtures.

---

## Expected artifacts
- `docs/datasets/unified_schema.md`
- `data/processed/<dataset>/artifacts.jsonl`
- `experiments/reports/<run_id>/results.json`
- `experiments/reports/<run_id>/pareto.json`
- `experiments/reports/<run_id>/paper_artifacts/paper_tables/pareto_table.csv`
- `experiments/reports/paper/cross_dataset_summary.csv`

---

## Implementation notes (recommended design)
### Canonical Artifact fields (minimum)
- `artifact_id` (stable, includes institution prefix)
- `type` (MISSION, COURSE, OUTCOME_REPORT, JOB_ROLE, SKILL, PAPER_ABSTRACT, …)
- `text` (normalized plain text used for embeddings)
- `metadata`:
  - `institution`: `MIT|UCB|ASU`
  - `unit`: department/school/program (if known)
  - `year` (for outcomes)
  - `url` / `source_url`
  - `license`
  - `retrieved_at`
  - `raw_hash`
  - optional: `tags`, `topic_codes`, `course_number`, `career_category`, `salary_stats`

### Outcomes normalization
Represent outcomes as a **separate table** keyed by:
- `institution`, `unit`, `cohort_year`, `category`, `value`, `source_url`, `raw_hash`

Then create a mapping layer:
- Berkeley FDS categories → canonical categories
- ASU career outcomes → canonical categories
- MIT outcomes (if used) → canonical categories

### Evaluator I/O contract (keep consistent)
- Input: `DatasetBundle` + `learner_profile_text` + config
- Output: `results.json`, `pareto.json`, `paper_artifacts/…`

---

## Prompt for Cursor Composer
> Implement KT1.4: create a unified schema doc, dataset registry (toy/mit/ucb/asu), dataset-specific configs, and a compare_datasets script that outputs cross_dataset_summary.csv.  
> Add fixtures if needed for CI.  
> Keep `pytest -q` and `python -m experiments.run dataset=toy ...` green.

## Prompt for Claude Code
> Read docs/issues/13_kt1_4_cross-university_alignment_(...).md.  
> Propose a plan (files/tests/commands).  
> Implement dataset registry + unified schema doc + compare_datasets script.  
> Run pytest and the toy + mit/ucb/asu stub runs and report output paths.
