# ADS Onboarding Overview (shared)

**Generated:** 2026-01-14

This repo is designed to turn ADS from a “blue-sky framework” into:
- a runnable prototype,
- a formal core,
- and reproducible experiments with paper-ready artifacts.

Key folders:
- `docs/issues/` — KT milestones (copy into tracker)
- `docs/policies/` — IJCAI compliance + reference integrity
- `experiments/` — Hydra configs + run entrypoint
- `packages/ads_core/` — core evaluation / lenses / constraints
- `packages/ads_api/` — FastAPI

Non-negotiables:
- Keep `dataset=toy` smoke test green.
- Open-data provenance for ingestion.
- No unverified references; use `tools/verify_bib.py`.

