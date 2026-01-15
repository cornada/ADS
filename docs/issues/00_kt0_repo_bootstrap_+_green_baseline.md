# [KT0] Repo bootstrap + green baseline

## Why
We need a stable, deterministic starting point: a repo that installs, runs the toy experiment, and has tests passing. This is the safety net for all future work.

## Scope
### In scope
- Initialize repo from template
- Add CLAUDE.md + Cursor rules
- Ensure toy experiment and pytest are green
- Add minimal CI-ready structure (optional)

### Out of scope
- Real ingestion from MIT/OpenAlex/O*NET
- Real UI dashboard

## Deliverables
- [ ] Repo boots: install + run + tests pass
- [ ] Added CLAUDE.md and .cursor/rules/ads.md
- [ ] README with quick start + smoke commands

## Acceptance criteria (must run and paste outputs/paths in PR)
- [ ] `pip install -e "[dev,exp]"  # or equivalent`
- [ ] `pytest -q`
- [ ] `python -m experiments.run dataset=toy embedding=stub lenses=identity`

## Expected artifacts (examples)
- `experiments/reports/<run_id>/config_resolved.yaml`
- `experiments/reports/<run_id>/paper_artifacts/paper_tables/pareto_table.csv`
- `experiments/reports/<run_id>/paper_artifacts/figures/pareto_scatter.png`

## Implementation notes
- Keep diffs small; add tests for new logic; keep toy run green.

## Prompt for Cursor Composer
> Bootstrap the repo, add CLAUDE.md + Cursor rules, run tests and a toy experiment; document outputs.

## Prompt for Claude Code
> Initialize the project: add CLAUDE.md and cursor rules, run pytest and the toy experiment, and report paths to generated paper_artifacts.
