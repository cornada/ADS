# [KT9] Paper-grade experiment pack + reproducibility bundle

## Why
Final milestone: produce paper-ready figures/tables with a one-command reproduce script on open data.

## Scope
### In scope
- Experiment matrix implementation (A1–A3 + B1 minimum)
- Automated report generator to produce final paper artifacts
- Reproduce script and documentation

### Out of scope
- Large-scale user study (optional later)

## Deliverables
- [ ] paper_tables/*.csv and figures/*.png for main results and ablations
- [ ] reproduce.md with exact commands and expected outputs
- [ ] archive/snapshot of datasets used (or deterministic fetch + hashes)

## Acceptance criteria (must run and paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `make reproduce_small  # add this target`
- [ ] `python -m experiments.run -m dataset=<open> embedding=<2 encoders> lenses=<3 modes>`

## Expected artifacts (examples)
- `experiments/reports/paper/paper_tables/*.csv`
- `experiments/reports/paper/figures/*.png`
- `experiments/reports/paper/reproduce.md`

## Implementation notes
- Keep outputs stable; pin model versions and dataset snapshots/hashes.

## Prompt for Cursor Composer
> Implement the experiment matrix (A/B) and a report builder that outputs paper tables/figures. Add make reproduce_small.

## Prompt for Claude Code
> Create a paper-grade reproducibility bundle: run the core experiments, generate paper artifacts, and write reproduce.md.
