# Reference Verification Protocol (Anti‑Hallucination)

## Why
Conference policies increasingly emphasize factual correctness.
We treat *every citation* as a potential failure mode, especially with GenAI assistance.

## Required standard (project)
- Every reference must be verifiable via at least one of:
  - DOI (Crossref),
  - arXiv ID,
  - OpenAlex work ID,
  - official proceedings page (for IJCAI/AAAI/etc.).

## Process
1) Add a new bib entry (prefer DOI + canonical metadata).
2) Run:
   - `python tools/verify_bib.py paper/references.bib --fail_on_unverified`
3) If the script proposes candidates, accept only after manual verification.
4) For “similar looking titles”, prefer DOI lookup to avoid wrong attribution.

## What counts as “verified”
- DOI resolves and metadata matches (title similarity + year +/- 1 + at least one author overlap), OR
- arXiv ID resolves and title matches.

## CI gate
- The repo must include a CI job that runs verify_bib on every PR touching `*.bib`.
- PR is blocked if unverified entries exist.

## Edge cases
- Workshop/tech report without DOI: must have stable URL + pdf + authors + year, and be manually approved (documented in `manual_overrides.yaml`).
- Datasets/code repos: cite the paper (preferred) or a DOI (Zenodo) if available.
