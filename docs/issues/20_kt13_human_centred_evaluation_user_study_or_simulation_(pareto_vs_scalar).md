# [KT13] Human-centred evaluation: user study OR agent-based simulation (Pareto vs scalar)

**Generated:** 2026-01-15  
**Owner:** Human-study lead + Engineering

## Goal
Add a human-centred evaluation component that matches IJCAI Human-Centred AI expectations:
compare a **scalar, black-box recommendation** vs a **Pareto/contestable dashboard**.

You may implement either:
- Option A: small user study (n=5–15) with volunteers (consent-based), OR
- Option B: agent-based simulation of students with heterogeneous preferences (if user study is infeasible).

Both options must produce an analyzable dataset and paper-ready summary.

## Why (reviewer risk addressed)
System papers in IJCAI-HAI often include:
- a user study, OR
- a rigorous offline evaluation with human-centred metrics (autonomy/contestability).

This KT creates the evaluation evidence for claims about contestability and autonomy.

## Inputs
- UI dashboard from KT8 (or CLI equivalent)
- Explainability/recourse from KT7
- Datasets from KT1.x (optional for simulation; can use fixtures)

## Deliverables (choose A or B; document which)
### Option A — User study (recommended if feasible)
Create:
- `docs/study/STUDY_PROTOCOL.md`
  - recruitment (volunteers), consent script, privacy policy
  - tasks: pick courses/pathways under scenario constraints
  - conditions: scalar vs pareto dashboard
- `docs/study/QUESTIONNAIRE.md`
  - autonomy (Likert), satisfaction, perceived control, trust, clarity
- `data/study/raw/` (NOT committed if sensitive; store locally)
- `data/study/processed/study_results.csv` (anonymized aggregates)
- `experiments/reports/paper/paper_tables/user_study_summary.csv`
- `experiments/reports/paper/paper_figures/user_study_bars.png`

Metrics (minimum):
- perceived autonomy score (mean ± std)
- perceived understanding of trade-offs
- diversity of chosen options (e.g., category entropy)
- time-to-decision (optional)

### Option B — Agent-based simulation (fallback)
Create:
- `packages/ads_core/sim/synthetic_students.py`
  - generate cohorts with different preference weights over objectives
- `experiments/scripts/sim_compare_scalar_vs_pareto.py`
- Outputs:
  - `experiments/reports/paper/paper_tables/sim_summary.csv`
  - `experiments/reports/paper/paper_figures/sim_diversity.png`

Metrics (minimum):
- diversity of selected pathways
- "zombie-risk" proxy rate (if defined)
- pareto coverage / regret vs scalar baseline

## Acceptance criteria
- [ ] One of the two options is fully implemented and documented
- [ ] All study/sim outputs are reproducible from scripts (or from a documented manual protocol)
- [ ] No PII is committed to the repo
- [ ] Paper-ready table + figure is produced under `experiments/reports/paper/`

## Prompt for Claude Code (copy/paste)
Implement KT13 using Option B (simulation) by default, unless study artifacts already exist.
Create scripts that generate paper-ready tables/figures into experiments/reports/paper/.
Add tests for determinism (fixed seed).
Do NOT commit any PII.

After implementation:
- run pytest -q
- run the sim script with seed=0
- list generated paper artifacts.
