# Mega Prompt for Claude Code — Next Phase bootstrap (KT10–KT15)

**Generated:** 2026-01-15

This is a copy/paste prompt intended to be used **inside Claude Code** (terminal agent) in the root of your ADS repo.

## Preconditions (human)
1) You are in the ADS repo root (same folder where `README.md`, `packages/`, `experiments/` exist).
2) Working tree is clean (commit or stash any local changes).
3) You have downloaded and extracted the tasks pack zip so that **one of** the following exists:
   - a folder `ADS_New_Tasks_ClaudeCode_Pack/` in the repo root, OR
   - a zip file `ADS_NextPhase_IJCAI2026_NewTasks_ClaudeCode.zip` in the repo root.

## IMPORTANT
- Do **NOT** draft final paper prose with LLMs. Only create **structure, templates, checklists, and TODO placeholders**.
- Keep `dataset=toy` smoke test green.

---

## Copy/paste prompt (use in Claude Code)

> You are working in an existing ADS repo. Your job: (A) merge the provided “Next Phase” tasks pack into this repo as a clean PR, and (B) create two follow-up PR branches for KT10 and KT11 that add only structure/templates/verification scaffolding (no final paper prose).  
>
> **Step 0 — sanity**
> 1. Run `git status` and confirm clean working tree. If not clean, STOP and tell me what is dirty.
> 2. Run `pytest -q` and `python -m experiments.run dataset=toy embedding=stub lenses=identity` and report results.
>
> **Step 1 — bring in the tasks pack**
> 1. Check if directory `ADS_New_Tasks_ClaudeCode_Pack/` exists in repo root.
>    - If not, but zip `ADS_NextPhase_IJCAI2026_NewTasks_ClaudeCode.zip` exists, unzip it into a temp dir and copy `ADS_New_Tasks_ClaudeCode_Pack/` into repo root.
>    - If neither exists, STOP and ask me to place the pack folder or zip into repo root.
> 2. Read `ADS_New_Tasks_ClaudeCode_Pack/MERGE_INTO_REPO.md` and follow it exactly.
> 3. Create a branch: `chore/next-phase-issues-pack`
> 4. Copy pack contents into repo root (docs/* additions). Do not overwrite unrelated repo files.
> 5. Ensure filenames are Windows-safe (no `:`).
> 6. Run `pytest -q` again.
> 7. Commit with message: `Add IJCAI next-phase issues pack (KT10–KT15)`
> 8. If `gh` is installed and `origin` remote exists:
>    - push branch and open a PR titled: `Next Phase tasks pack (KT10–KT15)`
>   Otherwise:
>    - print exact manual commands to push and create PR.
>
> **Step 2 — KT10 starter PR (structure only)**
> 1. Create branch from main: `kt10/paper-skeleton-claim-evidence`
> 2. Implement KT10 by creating real (non-template) files from the templates:
>    - `docs/paper/OUTLINE_IJCAI2026.md` (outline with section headings + TODO bullets only; no full prose)
>    - `docs/paper/CLAIM_EVIDENCE_MATRIX.csv` (fill with initial rows; leave TODO where unknown)
>    - `docs/paper/FORMAL_CORE_CHECKLIST.md` (checkbox list based on ADS formalization)
>    - `docs/paper/PAPER_ARTIFACTS_INDEX.md` (list expected figures/tables + source script paths)
> 3. Do not add any LaTeX paper text; only scaffolding.
> 4. Run `pytest -q` (should still pass).
> 5. Commit with message: `KT10: paper skeleton + claim→evidence matrix scaffolding`
> 6. Push and open PR `KT10: paper skeleton + claim→evidence matrix` (or print manual steps).
>
> **Step 3 — KT11 starter PR (related work coverage + verified bib gate)**
> 1. Create branch from main: `kt11/related-work-matrix-verified-bib`
> 2. Implement KT11 by creating:
>    - `docs/related_work/RELATED_WORK_MATRIX.csv` from template (add initial entries and placeholders)
>    - `docs/related_work/NOVELTY_MAP.md` from template (bullets only; no prose)
>    - ensure `tools/verify_bib.py` exists and add a `make verify_bib` target (or script) if missing
>    - add CI-friendly check target: generate `reports/reference_audit.json` when bib changes (documented)
> 3. Run:
>    - `python tools/verify_bib.py paper/references.bib --report reports/reference_audit.json --fail_on_unverified` (if network available)
>    - If network is not available: run in `--offline` mode if supported, else document how to run locally.
> 4. Run `pytest -q`.
> 5. Commit with message: `KT11: related work matrix + reference verification scaffolding`
> 6. Push and open PR `KT11: related work coverage + verified references` (or print manual steps).
>
> **Step 4 — report back**
> Provide:
> - list of created branches and whether PRs were opened
> - output paths for created docs files
> - command outputs (short)
> - any blockers

