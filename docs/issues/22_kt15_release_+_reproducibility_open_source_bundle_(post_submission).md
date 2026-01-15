# [KT15] Release & reproducibility: open-source bundle + artifact package (post-submission friendly)

**Generated:** 2026-01-15  
**Owner:** Engineering + Paper lead

## Goal
Prepare a clean, public-friendly release bundle that supports:
- reproducibility (at least "small" mode with fixtures),
- easy install/run,
- clean licensing & citation metadata,
- artifact packaging (tables/figures, scripts).

This is useful for:
- supplementary material,
- artifact evaluation (if applicable),
- later camera-ready release.

## Inputs
- Repro bundle (KT9) + final paper artifacts (KT12)
- LLM usage policy (KT2.0)
- dataset manifests (MIT/UCB/ASU/market) with links-only mode (no restricted raw data)

## Deliverables
### A) Repository metadata
- `LICENSE` (choose license and include)
- `CITATION.cff`
- `README.md` updated with:
  - quickstart (fixtures)
  - how to run paper artifact generation (key results)
- `REPRODUCE.md` (top-level)
- `SECURITY.md` (optional)
- `CODE_OF_CONDUCT.md` (optional)

### B) Reproduce commands
- `make reproduce_small` (or `python -m ...`) that runs:
  - unit tests
  - at least one end-to-end run on fixtures
  - regenerates at least one paper table/figure

### C) Release artifact bundle builder
- `tools/build_release_bundle.py`
Outputs:
- `dist/release_bundle.zip` containing:
  - source snapshot (or pointers)
  - reproduce instructions
  - paper artifacts (tables/figures)
  - dataset manifests

### D) Data policy clarity
- `docs/datasets/DATA_POLICY.md`
  - what is included in repo (fixtures/manifests)
  - what must be downloaded separately (raw data)
  - provenance + hashing rules

## Acceptance criteria
- [ ] `pytest -q`
- [ ] `make reproduce_small` succeeds on a clean machine (or documented manual steps)
- [ ] `dist/release_bundle.zip` is created and includes expected contents
- [ ] No PII or restricted raw data is included
- [ ] CITATION.cff is valid YAML

## Prompt for Claude Code (copy/paste)
Implement KT15:
- add LICENSE + CITATION.cff + REPRODUCE.md
- implement tools/build_release_bundle.py
- add make reproduce_small (or equivalent)
- create docs/datasets/DATA_POLICY.md

Run pytest -q and build the release bundle, then list its contents.
