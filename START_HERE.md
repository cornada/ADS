# START HERE — ADS project bootstrap (Claude Code)

**Generated:** 2026-01-14

This archive is a **full “start-from-zero” repository** prepared for working primarily with **Claude Code** as a terminal agent.

It includes:
- ADS codebase template (API + core + experiments),
- IJCAI-friendly workflow (issues, policies, reference verifier),
- `CLAUDE.md` with strict working agreement.

---

## 0) What success looks like (first 30 minutes)
You can run:
- `pytest -q`
- `python -m experiments.run dataset=toy embedding=stub lenses=identity`

…and you will get paper-ready artifacts in:
- `experiments/reports/<run_id>/paper_artifacts/...`

---

## 1) Install & run Claude Code
1) Install Claude Code (Anthropic) and authenticate.
2) In this repo folder, run:
```bash
claude
```

Claude Code will automatically load `CLAUDE.md` (project contract).

---

## 2) Local environment setup
Prereq: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,exp]"
# optional encoder
pip install -e ".[dev,exp,embed]"
```

---

## 3) Smoke test (must stay green always)
```bash
pytest -q
python -m experiments.run dataset=toy embedding=stub lenses=identity
```

Optional ablation sweep:
```bash
python -m experiments.run -m dataset=toy embedding=stub,sbert_allminilm lenses=identity,diagonal,learned
```

---

## 4) How to work (Claude Code workflow)
### 4.1 Pick the next issue template
Open `docs/issues/README.md` and choose the next KT file.

### 4.2 Prompt pattern (copy/paste into Claude Code)
> Read docs/issues/<KT file>.  
> Propose a plan (files/tests/commands).  
> Implement.  
> Run `pytest -q` and the acceptance commands.  
> Report output paths and a commit message.

---

## 5) References & LLM compliance (important)
- Policy docs: `docs/policies/`
- Reference checker: `python tools/verify_bib.py paper/references.bib --fail_on_unverified`

A minimal `paper/references.bib` is included so the tool can be tested.

---

## 6) Where to look in code
- Hydra experiments entrypoint: `experiments/run.py`
- Toy end-to-end pipeline: `packages/ads_core/pipeline/toy_pipeline.py`
- Report generation: `packages/ads_core/report/pareto_report.py`
- FastAPI app: `packages/ads_api/main.py`

---

## 7) First “real” deliverable after bootstrap
Implement KT1.1 (MIT curriculum ingestion) using open sources (catalog + OCW), with provenance fields.
Then rerun:
- `python -m experiments.run dataset=mit ...`

Keep the toy smoke test green and commit often.
