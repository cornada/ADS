# [KT1.2][UCB] UC Berkeley ingestion — outcomes (FDS) + major/college anchors

## Why
UC Berkeley is the strongest open source for **career outcome validation** (First Destination Survey and “Where Do Cal Grads Go?” dashboards).
This task builds an **outcomes corpus** that we can use to validate ADS claims like:
- “embedding-based market-fit correlates with real outcomes better than naive baselines”.

## Official sources (pin in a manifest)
- UC Berkeley OPA — First Destination Survey overview + links: https://opa.berkeley.edu/campus-surveys/survey-results-reporting-analysis/first-destination-survey
- OPA “Our Berkeley Data Digest” (lists dashboards, incl. FDS): https://opa.berkeley.edu/campus-data/our-berkeley-data-digest
- Berkeley Career Engagement — Where Do Cal Grads Go?: https://career.berkeley.edu/start-exploring/where-do-cal-grads-go/

## Scope
### In scope
- Build a pipeline that can ingest Berkeley outcomes **from a reproducible artifact**:
  - Preferred: officially downloadable CSV/PDF/HTML tables (if present).
  - Fallback: a manually exported CSV placed under `data/raw/ucb/fds/` with hashes recorded (still reproducible if documented).
- Produce a normalized dataset:
  - `data/processed/ucb/outcomes_major_level.csv` (or JSONL) with fields like:
    `year`, `college`, `major`, `employment_rate`, `grad_school_rate`, `median_salary` (if available), `n_respondents` (if available).
- Convert each row into a text artifact for embeddings:
  - `artifact_type="outcome_major"` with `text="Major X in College Y: employment ... salary ..."` + numeric metadata.

### Out of scope
- Any attempt to scrape non-public Tableau endpoints behind auth.
- PII or individual-level destinations.

## Deliverables
- [ ] `ads_core/ingest/ucb_outcomes.py` — parser/loader for Berkeley outcomes (CSV/PDF/HTML as available).
- [ ] `data/manifests/ucb_sources.yaml` — pinned source URLs + expected files.
- [ ] `data/processed/ucb/outcomes_major_level.csv` (or JSONL).
- [ ] `data/processed/ucb/artifacts.jsonl` including `outcome_major` artifacts with provenance.
- [ ] Tests: at minimum, parse a small fixture CSV in `tests/fixtures/ucb/...`.
- [ ] Docs: `docs/data_sources/ucb.md` including what the fallback/manual-export procedure is.

## Acceptance criteria (must run; paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `python -m ads_core.ingest.ucb --manifest data/manifests/ucb_sources.yaml --out_dir data/processed/ucb`
- [ ] `python -c "import pandas as pd; df=pd.read_csv('data/processed/ucb/outcomes_major_level.csv'); print(df.head(3))"`
- [ ] Optional if dataset integration included: `python -m experiments.run dataset=ucb embedding=stub lenses=identity`

## Expected artifacts
- `data/manifests/ucb_sources.yaml`
- `data/processed/ucb/outcomes_major_level.csv`
- `data/processed/ucb/artifacts.jsonl`

## Implementation notes
- Expect that some dashboards are JS-heavy; design the pipeline to accept “downloaded snapshot” inputs and record their hashes.
- Keep provenance strict: `source_url`, `retrieved_at`, `raw_hash`, and add `license` if the page states one (otherwise `license="unknown"`).

## Prompt for Cursor Composer
> Implement UC Berkeley outcomes ingestion (FDS) producing outcomes_major_level.csv and artifacts.jsonl.  
> Support a fallback path where a user drops exported CSVs into data/raw/ucb/fds/ and we record hashes.  
> Add tests with fixture CSV; keep toy run green.

## Prompt for Claude Code
> Implement ucb_outcomes ingestion per KT1.2.  
> Make it robust to manual CSV snapshots (hash recorded).  
> Run pytest and show the generated outcomes_major_level.csv head(3).
