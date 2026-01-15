# NEXT PHASE START HERE (ClaudeCode_WindowsSafe) — IJCAI 2026 packaging

**Generated:** 2026-01-15

This guide consolidates **all next-phase work** after completing all issues in the starter archive
(`ADS_Start_Zero_ClaudeCode_Full_WindowsSafe.zip`), so you do not have to collect steps from chat.

---

## What is "next phase"?
Your team has already implemented the core system (ingestion → embeddings → lenses → Pareto → ethics → explainability → UI)
and the open-data evaluation slices (MIT/UCB/ASU + market), plus compliance tooling.

Now the goal is to:
1) **freeze** experiments and artifacts (paper tables/figures),
2) produce a **submission-ready paper package** (outline → claims/evidence → related work → preflight checks),
3) optionally add a **human-centred evaluation** (user study or simulation),
4) prepare a **release/repro bundle**.

---

## Roles (recommended)
### You (project lead / paper lead)
- decide track strategy (HAI vs Demo vs AI&SocialGood)
- own paper claims and narrative
- ensure data sources are clean (no ToS risk, no PII in repo)
- final sign-off on results and references

### Colleague (engineering lead)
- implements code/tooling for KT10–KT15
- runs experiment matrix and generates artifacts
- automates preflight checks

---

## Order of execution (do not skip)
### Step 0 — Merge this pack into repo (1 PR)
1) Copy new files into repo:
   - `docs/issues/17_*` to `docs/issues/22_*`
   - `docs/onboarding/NEXT_PHASE_START_HERE.md`
2) Commit to branch: `chore/add-next-phase-issues`
3) Open PR and merge.

**Checkpoint CP0:** repo still green:
- `pytest -q`
- `python -m experiments.run dataset=toy embedding=stub lenses=identity`

---

### Step 1 — KT10: Paper skeleton + claim/evidence matrix
Work: `docs/issues/17_...`

**Checkpoint CP1:** You can answer:
- “What is our main claim?”
- “Which figure/table supports it?”
- “What would a reviewer attack?”

Outputs:
- OUTLINE_IJCAI2026.md
- CLAIM_EVIDENCE_MATRIX.csv
- FORMAL_CORE_CHECKLIST.md
- PAPER_ARTIFACTS_INDEX.md

---

### Step 2 — KT11: Related work matrix + verified references
Work: `docs/issues/18_...`

**Checkpoint CP2:** references are unkillable:
- `python tools/verify_bib.py paper/references.bib --report reports/reference_audit.json --fail_on_unverified`

Outputs:
- RELATED_WORK_MATRIX.csv
- NOVELTY_MAP.md
- updated references.bib (verified)

---

### Step 3 — KT12: Final experiment matrix + freeze paper artifacts
Work: `docs/issues/19_...`

**Checkpoint CP3:** paper artifacts exist under stable names:
- `experiments/reports/paper/paper_tables/...`
- `experiments/reports/paper/paper_figures/...`
- `experiments/reports/paper/PAPER_RESULTS_MANIFEST.json`

---

### Step 4 — KT13: Human-centred evaluation (optional but strong for HAI)
Work: `docs/issues/20_...`

Choose one:
- A) small volunteer user study (best, if feasible)
- B) simulation fallback (acceptable if well documented)

**Checkpoint CP4:** you have at least 1 table + 1 figure for HAI claims:
- user autonomy / contestability / diversity (or simulated proxies)

---

### Step 5 — KT14: Submission packaging + anonymization + preflight checks
Work: `docs/issues/21_...`

**Checkpoint CP5:** you can produce a submission folder:
- `dist/submission/` contains paper PDF + supplement + reproducibility files
- `python tools/preflight_submission.py --paper_dir paper` passes

---

### Step 6 — KT15: Release & reproducibility bundle
Work: `docs/issues/22_...`

**Checkpoint CP6:** you can create a clean release zip:
- `dist/release_bundle.zip`
- includes fixtures + manifests + reproduce instructions
- excludes PII and restricted raw data

---

## Practical "do not forget" list
- Keep Windows-safe filenames (no ":" characters).
- Do not commit raw outcomes exports if licensing/terms are unclear; commit fixtures + hashes + protocol instead.
- Never invent citations: if uncertain, mark CITE_TODO and verify later.
- Keep LLM usage log consistent with your policy (KT2.0).

---

## What you should do today (minimum)
1) Merge the task pack PR.
2) Start KT10 and KT11 in parallel:
   - paper outline + claim/evidence matrix
   - related-work matrix + verified references
3) Schedule KT12 run window (compute time) and freeze paper artifacts.

