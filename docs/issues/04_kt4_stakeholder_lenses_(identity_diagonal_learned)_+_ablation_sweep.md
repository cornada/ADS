# [KT4] Stakeholder lenses (identity/diagonal/learned) + ablation sweep

## Why
Lenses are the core ADS novelty: L_a(v) = W_a v. We must show an identity baseline, a simple lens, and a learned lens with ablations.

## Scope
### In scope
- Implement lens modes: identity, diagonal (MVP weights), learned diagonal (one-vs-rest)
- Wire lens mode into experiment config
- Add ablation sweep and report changes

### Out of scope
- Complex nonlinear lens models (keep for later)

## Deliverables
- [ ] ads_core.lenses implementations
- [ ] Hydra configs for 3 lens modes
- [ ] A sweep script that outputs multiple run folders

## Acceptance criteria (must run and paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `python -m experiments.run -m dataset=toy embedding=stub,sbert_allminilm lenses=identity,diagonal,learned`

## Expected artifacts (examples)
- `experiments/reports/<run_id>/paper_artifacts/figures/pareto_scatter.png (per run)`
- `A summary CSV comparing runs (optional)`

## Implementation notes
- Learned lens should be deterministic (seeded) and explained in docs.

## Prompt for Cursor Composer
> Add lens modes and a hydra multirun sweep (identity/diagonal/learned). Generate per-run paper artifacts.

## Prompt for Claude Code
> Implement identity/diagonal/learned lens modes and run a multirun sweep. Report the run directories and any differences in pareto sets.
