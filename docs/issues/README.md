# Issue templates (KT0–KT9)

These templates are designed to be copied into your tracker (GitHub Issues, Linear, Jira, Notion, etc.).
Each issue includes:
- Why (motivation)
- Deliverables
- Acceptance criteria (commands)
- Expected artifacts (paths)
- Prompt snippets for Cursor / Claude Code

## How to use
1) Pick the next KT milestone file.
2) Copy/paste into a new issue.
3) Assign to engineer + set priority.
4) Require the acceptance criteria outputs in the PR description.

## Global DoD for any issue
- Tests pass: `pytest -q`
- Smoke pass: `python -m experiments.run dataset=toy embedding=stub lenses=identity`
- Outputs are stored in `experiments/reports/<run_id>/...`


## University-specific KT1 extensions (MIT / UCB / ASU)
- KT1.1 MIT curriculum ingestion (catalog + OCW + mission)
- KT1.2 UC Berkeley outcomes ingestion (First Destination Survey)
- KT1.3 ASU ingestion (FDS instrument + career outcomes reports)

## Added university-specific ingestion milestones
- KT1.1 MIT curriculum ingestion
- KT1.2 UC Berkeley outcomes ingestion
- KT1.3 ASU outcomes/instrument ingestion
- KT1.4 Cross-university alignment (unified schema + registry + common evaluator)

- KT1.5 Cross-dataset outcome sanity-check (market fit vs aggregated outcomes)

- KT1.6 Stability under perturbations + fairness/representation stability (reviewer shield)

- KT2.0 IJCAI compliance: LLM usage policy + reference verifier automation
