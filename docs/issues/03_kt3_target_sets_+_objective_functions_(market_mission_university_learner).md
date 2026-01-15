# [KT3] Target sets + objective functions (market/mission/university/learner)

## Why
We must formalize vector-valued evaluation F(option) with explicit objective definitions and targets, producing interpretable trade-offs.

## Scope
### In scope
- Build target sets (centroids or clouds) from ingested corpora
- Define objective distances and normalization
- Output results.json with objectives per option

### Out of scope
- Full student modeling (KT) integration (separate milestone)

## Deliverables
- [ ] Target builders (centroid / set-of-points)
- [ ] Objective evaluation module
- [ ] Result serialization to JSON/CSV

## Acceptance criteria (must run and paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `python -m experiments.run dataset=toy embedding=stub lenses=identity`
- [ ] `python -m experiments.run dataset=<open> embedding=<...> lenses=identity  # once ingestion exists`

## Expected artifacts (examples)
- `experiments/reports/<run_id>/results.json`
- `experiments/reports/<run_id>/paper_artifacts/paper_tables/*.csv`

## Implementation notes
- Keep objective conventions consistent: lower-is-better distances (or document inversions).

## Prompt for Cursor Composer
> Implement target builders and objective evaluation producing results.json and paper_tables. Add tests for objective computations.

## Prompt for Claude Code
> Add explicit objective computations (market/mission/university/learner) and serialize results. Run toy experiment and show results.json snippet.
