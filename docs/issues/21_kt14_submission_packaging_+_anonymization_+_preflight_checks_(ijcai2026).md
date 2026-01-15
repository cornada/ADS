# [KT14] Submission packaging + anonymization + preflight checks (IJCAI 2026)

**Generated:** 2026-01-15  
**Owner:** Paper lead + Engineering (tooling)

## Goal
Create a submission-ready package:
- paper builds cleanly (PDF),
- anonymization is correct,
- compliance artifacts exist (LLM policy statement, verified references),
- supplementary/repro instructions are consistent.

## Why (reviewer/admin risk addressed)
Desk rejections happen for:
- formatting violations,
- anonymity leaks,
- broken references,
- missing disclosures.

This KT creates an automated "preflight" checklist.

## Inputs
- Paper outline + claim/evidence matrix (KT10)
- Verified bib workflow (KT11)
- Paper artifacts frozen (KT12)
- LLM policy + reference verifier (KT2.0)

## Deliverables
### A) Paper build scaffolding
- `paper/` directory containing:
  - `main.tex` (IJCAI template compatible)
  - `refs.bib` (or symlink to `paper/references.bib`)
  - `figures/` (symlinks or copied paper figures)
  - `tables/` (paper tables)
- `paper/BUILD.md` (how to compile)

> Do not commit proprietary templates if licensing forbids; instead commit a script that fetches the official template and patches in your files.

### B) Anonymization checklist + automated scan
- `docs/checklists/SUBMISSION_ANONYMIZATION_CHECKLIST.md`
- `tools/preflight_submission.py` (script)
Checks (minimum):
- no author names / affiliations / emails in tex/pdf metadata
- no acknowledgements (until camera-ready)
- no repo URLs or identifying paths in the main text
- bibliography is present + verify_bib passes
- figure filenames are anonymous

### C) Submission bundle script
- `tools/build_submission_bundle.py`
Outputs:
- `dist/submission/` with:
  - paper pdf
  - supplement pdf (optional)
  - reproducibility statement / reproduce.md
  - artifact tables/figures

## Acceptance criteria
- [ ] `python tools/preflight_submission.py --paper_dir paper` passes
- [ ] `latexmk -pdf paper/main.tex` (or equivalent) builds without errors
- [ ] references verification passes (`verify_bib`)
- [ ] `dist/submission/` exists and contains expected files

## Prompt for Claude Code (copy/paste)
Create tools/preflight_submission.py and docs/checklists/SUBMISSION_ANONYMIZATION_CHECKLIST.md.
Add a minimal paper/BUILD.md and dist/submission/ builder script.
Do NOT invent IJCAI template files; instead create placeholders and document where to download official template.
Run preflight and show a sample dist/submission/ tree.
