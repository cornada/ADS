# CLAUDE.md — ADS Project Working Agreement (Claude Code)
**Generated:** 2026-01-14

This file is loaded automatically by **Claude Code** when you run `claude` in this repo.
It defines the collaboration contract: commands, acceptance criteria, and non‑negotiable rules.

---

## 1) Project goals (what “done” means)
We are building **Agent‑Didactic Spaces (ADS)** as an **implemented system + formal core + experiments** (IJCAI‑grade).
The output must be:
- a runnable prototype (API + optional UI),
- reproducible experiments with paper-ready artifacts (tables/figures),
- open‑data provenance and verifiable references.

---

## 2) Hard rules (must not break)
### 2.1 Legal/ethics & data
- **No LinkedIn scraping or ToS‑fragile sources** for the main dataset.
- Only open/public sources with reproducible collection.
- Every artifact must store: `source_url`, `timestamp`, `raw_text_snapshot_hash`.
- No PII pipelines unless explicitly approved and consented.

### 2.2 Reproducibility (IJCAI‑friendly)
For every experiment run:
- write `config_resolved.yaml`, `seed`, `embedding.model_id`,
- write dataset hashes/provenance,
- outputs go to `experiments/reports/<run_id>/`.

### 2.3 Third-party code
- If a baseline repo **has no LICENSE** → treat as **reference only**; re-implement ideas.
- GPL/AGPL baselines must remain isolated (no code reuse into core).

### 2.4 Engineering hygiene
- Keep `toy` experiment runnable at all times (smoke test).
- Add tests for any non-trivial logic.
- Small diffs; one logical change per commit.

---

## 3) Standard commands (the agent must run)
### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,exp]"
# Optional: for sentence-transformers embeddings
pip install -e ".[dev,exp,embed]"
```

### Tests & lint
```bash
pytest -q
ruff check .
```

### Smoke runs (must stay green)
```bash
python -m experiments.run dataset=toy embedding=stub lenses=identity
python -m experiments.run -m dataset=toy embedding=stub,sbert_allminilm lenses=identity,diagonal,learned
```

### API
```bash
uvicorn ads_api.main:app --reload
```

---

## 4) How we work on a task (Claude Code workflow)
For every task/issue:
1) **Read the issue template** in `docs/issues/`.
2) Output a **plan**:
   - files to change
   - new tests
   - acceptance criteria commands
3) Implement changes.
4) Run the relevant commands (tests + smoke).
5) Summarize results:
   - what changed
   - how to reproduce
   - what remains TODO
6) Provide a suggested commit message (Conventional Commits style).

---

## 5) Output formatting (what to print back)
When reporting back after implementation, always include:
- ✅ commands executed
- ✅ paths to generated artifacts
- ✅ brief explanation of design choices
- ⚠️ any risks or TODOs

---

## 6) Paper/LLM policy reminders
- LLMs may be used as tools, but **not as authors**. Keep transparent notes about usage for acknowledgements and reproducibility.

---

## 7) Suggested branch/commit conventions
- Branch name: `kt<digit>-<short-slug>` (e.g., `kt1-ingest-mit`)
- Commit message: `feat(ingest): ...`, `fix(embed): ...`, `chore(ci): ...`


## Reference verification (mandatory)
- If you modify any `*.bib` file, run `python tools/verify_bib.py <bib> --fail_on_unverified` and attach `reports/reference_audit.json`.

## LLM usage policy (IJCAI 2026)
- Do NOT draft paper text. You may polish language. Any LLM used in methodology must be documented in Methods/Experiments.
