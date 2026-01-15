# PR Definition of Done (KT1.4 / KT1.5 / KT10–KT15)

Use this as a checklist in PR descriptions.

## Always
- [ ] Windows-safe filenames (no ':')
- [ ] `pytest -q`
- [ ] `python -m experiments.run dataset=toy embedding=stub lenses=identity`
- [ ] Provenance fields validated by tests (`source_url`, `retrieved_at`, `raw_hash`, `license`, `institution`)
- [ ] `data_hash`, `code_hash`, `generated_at` included in paper artifact CSVs

## KT1.4 (alignment)
- [ ] registry loads datasets offline (fixtures)
- [ ] `compare_datasets.py` generates `experiments/reports/paper/cross_dataset_summary.csv`

## KT1.5 (outcome sanity)
- [ ] `outcome_sanity.py` generates:
  - `experiments/reports/paper/cross_dataset_outcome_sanity.csv`
  - `experiments/reports/paper/figures/outcome_sanity_scatter.png`
- [ ] baseline `B0_major_name_only` included
- [ ] no leakage (outcomes not used to form embeddings)

## KT11 (references)
- [ ] `python tools/verify_bib.py paper/references.bib --report reports/reference_audit.json --fail_on_unverified` passes

## KT14 (submission)
- [ ] `python tools/preflight_submission.py --paper_dir paper` passes
- [ ] `dist/submission/` is produced

## KT15 (release)
- [ ] `dist/release_bundle.zip` is produced
- [ ] no PII / restricted raw data in bundle
