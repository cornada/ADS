# [KT5] Pareto engine + short pathway planning + contestability toggles

## Why
ADS must avoid scalar scoring: show Pareto-frontier and allow the user to change objectives/constraints to see trade-offs (contestability).

## Scope
### In scope
- Pareto nondominated sorting over selected objectives
- Short pathway planning (depth 2–3) with beam search
- API/config toggles for objectives and constraints

### Out of scope
- Full prerequisite constraint solving (optional later)

## Deliverables
- [ ] Pareto outputs for options and pathways
- [ ] Config/API parameters to switch objective sets and tau constraints
- [ ] Figures: Pareto scatter + radar examples (optional)

## Acceptance criteria (must run and paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `python -m experiments.run dataset=toy embedding=stub lenses=identity`
- [ ] `python -m experiments.run dataset=toy embedding=stub lenses=identity constraints.autonomy_drift_tau=0.1`

## Expected artifacts (examples)
- `experiments/reports/<run_id>/pareto.json`
- `experiments/reports/<run_id>/paper_artifacts/figures/pareto_scatter.png`

## Implementation notes
- Keep a clean separation: ads_core library vs experiments harness vs API.

## Prompt for Cursor Composer
> Implement Pareto-front computation and short pathway planning; add toggles for objective sets and constraints; show that changing tau changes the Pareto set.

## Prompt for Claude Code
> Add Pareto engine + pathway planner (depth 2–3). Run two configs with different tau and show that Pareto outputs differ.
