# [KT1.1][MIT] MIT ingestion — mission + course catalog + OCW curriculum corpus

## Why
MIT is the best single-university starting point for ADS **curriculum semantics**, because it offers:
- structured official course descriptions (catalog/subject listing), and
- rich open course materials via **MIT OpenCourseWare (OCW)**.

This task builds an **MIT curriculum corpus** that is legally clean and reproducible, and that feeds directly into ADS embeddings and Pareto evaluation.

## Official sources (pin in a manifest)
- MIT Course Catalog — Subjects: https://catalog.mit.edu/subjects/
- Subjects PDF snapshot (useful for deterministic parsing): https://catalog.mit.edu/subjects/subjects.pdf
- MIT Subject Listing & Schedule: https://student.mit.edu/catalog/index.cgi
- MIT OpenCourseWare (OCW): https://ocw.mit.edu/
- OCW terms/license (CC BY-NC-SA 4.0): https://ocw.mit.edu/pages/privacy-and-terms-of-use/

## Scope
### In scope
- Choose **one department/program** as the initial slice (recommended: Course 6 / EECS-style; but configurable).
- Ingest:
  1) **Mission texts** (MIT-level + department/program mission page(s)).
  2) **Course catalog descriptions** for the chosen department (title, number, description, prerequisites, units).
  3) **OCW course pages** for 10–30 courses (overview + syllabus-like text; avoid downloading huge binaries unless needed).
- Normalize into `Artifact` JSONL with required provenance:
  - `source_url`, `retrieved_at`, `raw_text_hash`, `license`, `institution="mit"`, `artifact_type`.
- Add local fixtures for parsers (HTML/PDF samples) and unit tests.

### Out of scope
- Scraping non-public student data.
- Any ToS-fragile crawling (no logins, no robot bypassing).
- Mass downloading of OCW video/large files (keep MVP lean).

## Deliverables
- [ ] `ads_core/ingest/mit_catalog.py` — fetch+parse MIT catalog subjects (HTML and/or PDF).
- [ ] `ads_core/ingest/mit_ocw.py` — fetch+parse OCW course pages (manifest-driven).
- [ ] `data/manifests/mit_sources.yaml` — pinned list of URLs / department selection / course list.
- [ ] `data/processed/mit/artifacts.jsonl` (+ optional `provenance.jsonl` if you keep it separate).
- [ ] Unit tests for parsers (use `tests/fixtures/mit/...`).
- [ ] Docs: `docs/data_sources/mit.md` describing what we collect and the license implications.

## Acceptance criteria (must run; paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `python -m ads_core.ingest.mit --manifest data/manifests/mit_sources.yaml --out data/processed/mit/artifacts.jsonl`
- [ ] `python -c "import json; print(sum(1 for _ in open('data/processed/mit/artifacts.jsonl')))"`  # non-zero
- [ ] If dataset integration is included: `python -m experiments.run dataset=mit embedding=stub lenses=identity`

## Expected artifacts
- `data/manifests/mit_sources.yaml`
- `data/processed/mit/artifacts.jsonl`
- `data/processed/mit/provenance.jsonl` (optional)
- `experiments/reports/<run_id>/...` (if wired into experiments)

## Implementation notes
- Keep ingestion deterministic: prefer **manifest-driven** selection over crawling “everything”.
- Store only what we need; if license is CC BY-NC-SA, we can store text locally for research, but be careful about redistributing large raw snapshots inside the repo.
- Record `license="CC BY-NC-SA 4.0"` on OCW-derived artifacts.

## Prompt for Cursor Composer
> Implement MIT ingestion (catalog + OCW + mission) using a manifest-driven approach.  
> Output artifacts.jsonl with provenance + license fields. Add parser fixtures + unit tests. Keep toy run green.

## Prompt for Claude Code
> Read docs/issues/10_kt1_1_mit_curriculum_ingestion...  
> Propose a plan (files, tests, commands).  
> Implement manifest-driven MIT catalog + OCW ingestion into artifacts.jsonl with provenance+license.  
> Run pytest and the ingestion command; show sample artifacts.
