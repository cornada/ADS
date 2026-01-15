# [KT1] Open-data ingestion (MVP: missions + courses + market roles)

## Why
IJCAI reviewers will not accept a purely conceptual framework. We need open, reproducible data to build the ADS space (mission ↔ courses ↔ market) with clean provenance.

## Scope
### In scope
- Implement ingestion connectors (at least 2 sources): course catalog/OCW-style + O*NET/ESCO-style roles
- Normalize into Artifact schema (JSONL)
- Record provenance fields (source_url, timestamp, raw_text_hash, optional snapshot)

### Out of scope
- Student PII ingestion
- LinkedIn scraping

## Deliverables
- [ ] Ingestion scripts in ads_core.ingest
- [ ] Generated artifacts.jsonl from a real open source OR a pinned snapshot file
- [ ] Unit tests for normalization/provenance

## Acceptance criteria (must run and paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `python -m experiments.run dataset=toy embedding=stub lenses=identity`
- [ ] `python -m experiments.ingest --config experiments/conf/dataset/<new>.yaml  # if implemented`

## Expected artifacts (examples)
- `data/processed/artifacts.jsonl`
- `data/processed/provenance.jsonl (or provenance fields inside artifacts)`
- `experiments/reports/<run_id>/config_resolved.yaml`

## Implementation notes
- Prefer MIT/OCW-style for content richness and O*NET/ESCO for market anchors. Keep ingestion reproducible (snapshot or deterministic download).

## Prompt for Cursor Composer
> Implement open-data ingestion into artifacts.jsonl with provenance fields; add tests; keep toy run green.

## Prompt for Claude Code
> Add ingestion pipeline that outputs artifacts.jsonl with provenance. Add tests. Run pytest and show a sample artifact record.
