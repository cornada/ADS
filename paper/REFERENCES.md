# References workflow (team)

## Add a new citation
1) Prefer DOI-based BibTeX from publisher/Crossref.
2) Add entry to `paper/references.bib`.
3) Run:
   - `python tools/verify_bib.py paper/references.bib --fail_on_unverified`
4) If unverified:
   - add DOI/arXiv ID
   - or add a manual override (rare; document why)

## Golden rule
If we cannot verify the reference automatically, we cannot cite it.
