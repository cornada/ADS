# [KT6] Ethics/Fairness layer + stability evaluation

## Why
To be IJCAI-strong, ADS must address autonomy/fairness and report stability under perturbations (not just utility).

## Scope
### In scope
- Formalize autonomy-drift (trajectory functional or proxy)
- Implement constraints/alerts in evaluation output
- Add a perturbation test to measure stability (e.g., small noise in embeddings, resampling targets)

### Out of scope
- Full demographic fairness without data (do minimal open-data version)

## Deliverables
- [ ] Constraint evaluation results (feasible/violations)
- [ ] Stability report (Jaccard of Pareto sets under perturbations)
- [ ] Paper-ready plot/table

## Acceptance criteria (must run and paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `python -m experiments.run dataset=toy embedding=stub lenses=identity constraints.autonomy_drift_tau=0.25`
- [ ] `python -m experiments.run dataset=toy embedding=stub lenses=identity +stability.n_runs=5  # if implemented`

## Expected artifacts (examples)
- `experiments/reports/<run_id>/paper_artifacts/paper_tables/stability.csv (optional)`
- `experiments/reports/<run_id>/paper_artifacts/figures/autonomy_tradeoff.png (optional)`

## Implementation notes
- Keep terminology neutral/professional; avoid loaded terms in code and paper.

## Prompt for Cursor Composer
> Add ethics constraint outputs and a stability evaluation (perturbation) producing a table/plot. Keep runs deterministic via seeds.

## Prompt for Claude Code
> Implement ethics constraints and stability evaluation. Run toy experiment with stability=on and output stability metrics.
