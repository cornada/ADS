# [KT1.3][ASU] Arizona State University ingestion — FDS instrument + outcomes reports

## Why
ASU is useful for:
- publicly available **First Destination Survey (FDS) instrument/logic**, and
- public **career outcomes reports** (even if some are older, they are still usable as a validation slice).

This task builds an ASU “survey + outcomes” corpus that:
- anchors our evaluation methodology (survey structure),
- provides extra outcomes evidence beyond Berkeley.

## Official sources (pin in a manifest)
- ASU UOEEE — First Destination Survey (FDS) Survey Preview (PDF): https://uoeee.asu.edu/sites/g/files/litvpz631/files/docs/Survey/FDS%20Survey%20Preview.pdf
- ASU Provost — Undergraduate Career Outcomes report (PDF): https://provost.asu.edu/sites/g/files/litvpz671/files/page/2550/career_outcomes_1516.pdf
- ASU UOEEE — Reports landing page: https://uoeee.asu.edu/reports

## Scope
### In scope
- Ingest the FDS **survey logic** PDF into a structured JSON schema:
  - `data/processed/asu/fds_schema.json` (questions, branches, answer options).
- Ingest at least one **career outcomes** PDF into a structured table:
  - `data/processed/asu/outcomes_summary.csv`
- Create text artifacts for embeddings:
  - `artifact_type="outcome_summary"` and/or `artifact_type="survey_schema"` (a short text summary per section).

### Out of scope
- Any reports that require portal access / credentials.
- Individual-level outcome data.

## Deliverables
- [ ] `ads_core/ingest/asu_fds.py` — extract survey structure from PDF.
- [ ] `ads_core/ingest/asu_outcomes.py` — extract outcomes summary tables from PDF(s).
- [ ] `data/manifests/asu_sources.yaml`
- [ ] `data/processed/asu/fds_schema.json`
- [ ] `data/processed/asu/outcomes_summary.csv`
- [ ] `data/processed/asu/artifacts.jsonl` with provenance.
- [ ] Tests with fixtures (store small PDFs or extracted text fixtures in `tests/fixtures/asu/...`).

## Acceptance criteria (must run; paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `python -m ads_core.ingest.asu --manifest data/manifests/asu_sources.yaml --out_dir data/processed/asu`
- [ ] `python -c "import json; print(list(json.load(open('data/processed/asu/fds_schema.json')).keys())[:10])"`
- [ ] `python -c "import pandas as pd; print(pd.read_csv('data/processed/asu/outcomes_summary.csv').head(3))"`

## Expected artifacts
- `data/manifests/asu_sources.yaml`
- `data/processed/asu/fds_schema.json`
- `data/processed/asu/outcomes_summary.csv`
- `data/processed/asu/artifacts.jsonl`

## Implementation notes
- PDF parsing can be messy. For MVP: a “good enough” extraction that preserves the branching structure + key outcome numbers is sufficient.
- Keep provenance strict: store `source_url`, `retrieved_at`, `raw_hash`, and any stated license/usage notes.

## Prompt for Cursor Composer
> Implement ASU ingestion: parse FDS Survey Preview PDF into fds_schema.json and parse career_outcomes_1516.pdf into outcomes_summary.csv.  
> Create artifacts.jsonl with provenance. Add tests with fixtures.

## Prompt for Claude Code
> Build ASU ingestion for FDS + outcomes PDFs.  
> Ensure outputs are structured and deterministic.  
> Run pytest and show heads/snippets of the produced JSON/CSV.
