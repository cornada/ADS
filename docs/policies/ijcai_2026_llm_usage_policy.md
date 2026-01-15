# IJCAI‑ECAI 2026: Internal LLM Usage Policy (Team Contract)

**Goal:** ensure we never get desk‑rejected for improper GenAI usage and that our submission is compliant and defensible.

This policy is aligned with the IJCAI‑ECAI 2026 public rules (Main Track CFP + Submissions FAQ + track CFPs).
If official rules change, this file MUST be updated.

---

## 0) Core rules we follow (non‑negotiable)
1) **LLMs are tools, not authors.** We never list any GenAI system as an author.
2) **Humans are accountable** for every factual claim, equation, experiment, and reference.
3) **No LLM‑written paper text**: we do NOT use LLMs to draft the submission narrative.
   - Allowed: style/language polishing ONLY.
4) If any LLM output is part of the *methodology* or *experimental analysis*, it must be **explicitly documented in the paper** (Methods/Experimental Setup), not hidden.

---

## 1) Allowed LLM usage (safe / compliant)
### A) Writing & editing
✅ Allowed:
- Grammar/style polishing of text that was originally written by a human author.
- Rephrasing for clarity while keeping meaning unchanged.

❌ Not allowed:
- Asking an LLM to “write the introduction / related work / experiments section”.
- Asking for “a full paper draft” and then editing it lightly.

### B) Research & engineering
✅ Allowed (with constraints):
- Code assistance (boilerplate, tests) — must be reviewed, tested, and license‑checked.
- Summaries of papers we already have, to speed up internal reading (but the final write‑up is human).
- Data extraction where the LLM is part of the method (e.g., LLM grading / rubric application), **only if** we document:
  - the exact model/tool,
  - prompts or rubric,
  - sampling/temperature,
  - QA (spot checks / failure analysis),
  - and limitations.

### C) Figures / tables
✅ Allowed:
- LLM can help generate captions or descriptions, but we must verify all numbers and labels.

---

## 2) Mandatory documentation (our internal provenance)
We maintain an internal `docs/policies/llm_usage_log.md` with entries per PR:

Required fields:
- Date, author
- Tool name + model/version (if known)
- What was used for (e.g., language polish, code scaffolding, data labeling)
- What human checks were performed (tests run, manual verification, spot‑check size)
- Any sensitive data used? (must be “NO” for paper submission work)

---

## 3) What to put in the paper (submission vs camera‑ready)
### Submission (double blind)
- Do NOT include acknowledgements in the PDF.
- If LLMs are used **as part of the method/analysis**, describe them in Methods/Experimental Setup:
  - “We used [tool/model] to …; parameters …; evaluation …; limitations …”

### Camera‑ready (after acceptance)
- Add an acknowledgements paragraph summarizing permitted LLM usage (style polishing, etc.), if needed.

---

## 4) Reference integrity (GenAI risk hot‑spot)
All references MUST be real and verifiable.
We enforce this via an automated `verify_bib.py` tool + CI gate:
- Each bib entry must have DOI/arXiv ID or a Crossref/OpenAlex match.
- Any unverified entry blocks merge.

---

## 5) Escalation rules
If anyone believes a section is “too AI‑written”, we:
1) revert to a human‑written version,
2) re‑run checks,
3) log corrective action in the usage log.

---

## 6) Links to official rules (pin these in README / paper tracker)
- Main track CFP: https://2026.ijcai.org/ijcai-ecai-2026-call-for-papers-main-track/
- Submissions FAQ: https://2026.ijcai.org/submissions-faq/
- Demo CFP (has LLM clause too): https://2026.ijcai.org/ijcai-ecai-2026-call-for-papers-demos/
- Survey track CFP (explicit desk‑reject for factual errors including references): https://2026.ijcai.org/ijcai-ecai-2026-call-for-papers-survey/
