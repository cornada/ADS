# [KT2.0] IJCAI‑ECAI 2026 Compliance: LLM usage policy + reference verifier automation

## Why
Two known “paper killers” for IJCAI‑style reviewing (and desk‑rejection filters):
1) **Improper GenAI usage** (paper appears to be written by an LLM, or GenAI used without accountability).
2) **Factual errors in references** (wrong/made‑up citations, incorrect metadata).

We need a *hard* engineering gate:
- internal LLM usage contract + log,
- automated reference verification (Crossref/OpenAlex/arXiv),
- CI/pre‑commit enforcement.

This directly addresses the post‑AAMAS risk: *no hallucinated references*.

---

## Scope
### In scope
- Add / finalize internal policy docs:
  - `docs/policies/ijcai_2026_llm_usage_policy.md`
  - `docs/policies/llm_usage_log.md`
  - `docs/policies/reference_verification_protocol.md`
- Implement `tools/verify_bib.py`:
  - parses `.bib`
  - verifies DOI/arXiv
  - produces a JSON audit report
  - fails CI if unverified refs exist
- Add CI gate (GitHub Actions / whatever):
  - on PR, run `python tools/verify_bib.py paper/references.bib --fail_on_unverified`
- Add a “paper‑build safety checklist”:
  - no acknowledgements in submission PDF
  - anonymization / metadata stripping
  - references verified

### Out of scope
- Full latex build pipeline (can be separate KT)
- Full plagiarism detection tooling (optional add‑on)

---

## Deliverables
- [ ] Policy docs in `docs/policies/` finalized and referenced from `README.md`
- [ ] `tools/verify_bib.py` runnable and documented
- [ ] `reports/reference_audit.json` sample report committed from a toy bib (fixtures)
- [ ] CI job / pre-commit hook that runs reference verification
- [ ] `paper/REFERENCES.md` (or similar) describing how to add new citations safely

---

## Acceptance criteria (must run and paste outputs in PR)
- [ ] `python tools/verify_bib.py paper/references.bib --report reports/reference_audit.json --fail_on_unverified --sleep 0.2`
- [ ] If paper source is in repo: `latexmk -pdf paper/main.tex` (or equivalent) produces a PDF with:
  - no acknowledgements section in submission mode
  - no author names/affiliations
- [ ] `pytest -q` still passes

---

## Notes / Implementation guidance
### 1) How to avoid false positives in reference verification
- Require DOI for almost everything.
- For arXiv, store `eprint` + `archivePrefix=arXiv`.
- For proceedings, add DOI when available.

### 2) What if Crossref/OpenAlex disagree?
- Prefer DOI lookup (Crossref) if DOI exists.
- Otherwise manually approve in `manual_overrides.yaml` with an explicit note + stable URL.

### 3) LLM disclosure strategy for IJCAI‑ECAI 2026
- Submission is double‑blind and forbids acknowledgements.
- If LLM is part of the *method*, document it in Methods/Experiments.
- For language polishing only, keep an internal log; optionally disclose in camera‑ready acknowledgements.

---

## Prompt for Cursor Composer
> Implement KT2.0: add policy docs, implement tools/verify_bib.py, integrate reference verification into CI or pre-commit, and add a short “how to add citations safely” doc. Make sure the verifier fails on unknown references. Include a toy fixtures bib + sample report.

## Prompt for Claude Code
> Read docs/issues/16_kt2_0_llm_usage_policy_+_reference_verifier_automation_(...).md. Propose plan (files/tests/commands). Implement verify_bib tool + docs and wire it into CI. Run verifier on fixtures and report output paths.
