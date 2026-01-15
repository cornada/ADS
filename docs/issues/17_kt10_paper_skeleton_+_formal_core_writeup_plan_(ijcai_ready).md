# [KT10] Paper skeleton + formal core write-up plan (IJCAI-ready)

**Generated:** 2026-01-15  
**Owner:** Paper lead (human) + Claude Code (assist for structure, not authorship)

## Goal
Turn the current project from "implemented system" into a **submission-ready IJCAI paper** by producing:
1) a paper outline aligned with the implemented system & experiments,
2) a "claim → evidence" matrix that maps every paper claim to a concrete artifact (table/figure/ablation),
3) a formal-core checklist (math definitions to include, consistent notation).

> Important: LLM tools may help with **structure and wording suggestions**, but the scientific content, claims, and decisions must be **human-authored and verified** (per IJCAI/LLM policy + good practice).

## Why (reviewer risk addressed)
Reviewers reject "blue-sky" work when:
- contributions are not crisply stated,
- math core is hand-wavy,
- experiments don't match claims,
- figures are pretty but not tied to RQs.

This KT creates the scaffolding so nothing is missed later.

## Inputs
- Implemented pipeline + reports from KT0–KT9 and KT1.1–KT1.6
- Compliance tools from KT2.0 (reference verifier + LLM usage policy)
- Existing draft text / draft variants (if any)

## Deliverables
Create these files in-repo:

### A) Paper outline (living document)
- `docs/paper/OUTLINE_IJCAI2026.md`
  - proposed track positioning: Human-Centred AI vs Demo vs AI & Social Good
  - section skeleton (Intro, Related Work, Method, System, Experiments, Limitations, Ethics)

### B) Claim → Evidence matrix
- `docs/paper/CLAIM_EVIDENCE_MATRIX.csv`
  Columns (minimum):
  - claim_id
  - claim_text (short, human-verified)
  - required_evidence (table/fig/ablation/user-study)
  - produced_by (script path)
  - artifact_path
  - status (todo/partial/done)
  - risks (what could break / what reviewer might attack)

### C) Formal core checklist
- `docs/paper/FORMAL_CORE_CHECKLIST.md`
  Must cover (minimum):
  - embedding space E ⊂ R^d and representation r(x)
  - stakeholder lenses L_a(v) (linear/diagonal/learned), metrics d_a
  - vector evaluation F(i) components (market, mission, university, learner, constraints)
  - Pareto frontier definition + contestability toggles
  - stability metrics (perturbations) + representation stability
  - explainability/recourse formal output (counterfactual what-if)

### D) Paper artifact index
- `docs/paper/PAPER_ARTIFACTS_INDEX.md`
  - list expected final output files under `experiments/reports/paper/`
  - naming conventions (Windows-safe)

## Acceptance criteria
- [ ] The outline covers all implemented components and matches your target IJCAI track
- [ ] Every major claim has an evidence row in the matrix (no orphan claims)
- [ ] Formal core checklist has zero "TBD" for definitions/notation (only optional extensions can be TBD)
- [ ] The artifact index references real (or planned) script outputs with paths

## Prompt for Claude Code (copy/paste)
Read the existing repo docs/issues/ and summarize what is implemented and what evidence artifacts exist.
Create:
- docs/paper/OUTLINE_IJCAI2026.md
- docs/paper/CLAIM_EVIDENCE_MATRIX.csv (seed with best-effort rows; mark unknown as TODO)
- docs/paper/FORMAL_CORE_CHECKLIST.md
- docs/paper/PAPER_ARTIFACTS_INDEX.md

Constraints:
- Do NOT invent citations. If a claim needs a citation, write "CITE_TODO".
- Do NOT write the final paper; only outline + checklists + matrix.

After writing files:
- run `pytest -q`
- print the first 10 lines of CLAIM_EVIDENCE_MATRIX.csv in the PR description.

## Notes
- Keep all file names Windows-safe (no ":" characters).
- Keep the matrix honest: if evidence is weak, mark it.
