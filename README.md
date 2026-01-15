# Agent‑Didactic Spaces (ADS) — Repo Template (v2)

**Generated:** 2026-01-14

This template is designed to be **handoff‑ready** for a colleague implementing:
- ADS core library (embeddings → lenses → multi‑objective eval → Pareto → constraints),
- FastAPI backend,
- experiment harness with **Hydra/OmegaConf** + automated ablation sweeps,
- report generation (paper‑ready figures/tables).

> MVP runs end‑to‑end on a built‑in **toy open dataset** (no internet, no PII).  
> Real ingestion connectors (MIT/OCW/OpenAlex/O*NET) are stubbed but scaffolded.

---

## Quick start (local dev)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -U pip

# Install core + API + experiment extras
pip install -e ".[dev,exp]"

# Run tests
./scripts/test.sh           # On Windows: .\scripts\test.ps1

# Run API
uvicorn ads_api.main:app --reload
```

Open:
- API docs: http://localhost:8000/docs

### Run tests

```bash
# Linux/macOS
./scripts/test.sh

# Windows PowerShell
.\scripts\test.ps1

# Or directly with PYTHONPATH
PYTHONPATH=packages pytest -q
```

### Run a single experiment (toy dataset)
```bash
python -m experiments.run dataset=toy embedding=stub lenses=identity
```

### Run ablation sweep (2 encoders × 3 lens modes)
```bash
python -m experiments.run -m \
  dataset=toy \
  embedding=stub,sbert_allminilm \
  lenses=identity,diagonal,learned
```

Outputs are written to:
- `experiments/reports/<run_id>/` (JSON results, CSV tables, PNG figures)

---

## What you get (v2 upgrades)
- ✅ Hydra/OmegaConf config system (`experiments/conf/`)
- ✅ Multi-run sweeps + deterministic output folders
- ✅ Report builder (Pareto plots + summary tables)
- ✅ Two encoders (stub + sentence-transformers)
- ✅ Three lens modes (identity + diagonal + learned(diagonal via logistic regression))
- ✅ Toy dataset generator for smoke tests and CI

---

## Docs
- `docs/ADS_Implementation_Spec.md`
- `docs/ADS_Experiment_Plan_and_Matrix.md`
- `docs/Reproducibility_Checklist.md`

---

## Licensing / baselines policy
See `THIRD_PARTY_NOTICES.md`.  
Do not vendor code from repos with missing license; use as reference only.
