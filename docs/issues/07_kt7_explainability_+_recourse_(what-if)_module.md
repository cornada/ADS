# [KT7] Explainability + recourse (what-if) module

## Why
Contestability requires actionable explanations: why an option is on Pareto, and what minimal change would move to another trade-off point.

## Scope
### In scope
- Evidence-based explanation (nearest neighbors to targets per objective)
- Heuristic recourse: suggest a bridging option that improves a selected objective with minimal regressions
- API endpoint to request explanation/recourse

### Out of scope
- RobustX integration (optional follow-up issue)

## Deliverables
- [ ] Explain payload for a chosen option
- [ ] Recourse suggestions payload
- [ ] Small demo notebook or CLI showing before/after

## Acceptance criteria (must run and paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `python -m experiments.run dataset=toy embedding=stub lenses=identity`
- [ ] `python -c "# call explain/recourse function"  # provide exact command in PR`

## Expected artifacts (examples)
- `experiments/reports/<run_id>/paper_artifacts/figures/recourse_example.png (optional)`
- `logs showing explanation output`

## Implementation notes
- Keep outputs human-readable; this is a key demo feature.

## Prompt for Cursor Composer
> Implement explanation (nearest neighbor evidence) and heuristic recourse. Add an API route for explain/recourse and a small demo.

## Prompt for Claude Code
> Add explainability outputs and a simple recourse heuristic. Demonstrate with toy data: print option, explanation, and recourse suggestion.
