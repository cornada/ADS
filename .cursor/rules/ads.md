# Cursor Rules for ADS Project

## Project Overview
Agent-Didactic Spaces (ADS) - an implemented system + formal core + experiments for IJCAI submission.

## Key Commands
```bash
# Setup
pip install -e ".[dev,exp]"

# Tests
pytest -q
ruff check .

# Smoke runs (must stay green)
python -m experiments.run dataset=toy embedding=stub lenses=identity

# API
uvicorn ads_api.main:app --reload
```

## Hard Rules
1. **No LinkedIn scraping** or ToS-fragile sources
2. **Reproducibility**: Every experiment writes config_resolved.yaml, seed, model_id, dataset hashes
3. **Outputs**: `experiments/reports/<run_id>/`
4. **No PII** unless explicitly approved
5. **Keep toy experiment green** at all times

## Code Style
- Line length: 100 (ruff)
- Python >=3.10
- Small diffs; one logical change per commit
- Conventional Commits: `feat(scope): ...`, `fix(scope): ...`

## Package Structure
- `packages/ads_core/` - Core library (embeddings, lenses, eval, pareto)
- `packages/ads_api/` - FastAPI backend
- `experiments/` - Hydra configs and run harness
- `paper/` - References and paper artifacts

## Testing
- Add tests for any non-trivial logic
- Tests in `tests/` directory
- Use pytest fixtures for shared setup

## Reference Verification
If modifying `*.bib` files, run:
```bash
python tools/verify_bib.py <bib> --fail_on_unverified
```

## LLM Usage Policy (IJCAI 2026)
- LLMs may be used as tools, not authors
- Document any LLM used in methodology
- Do NOT draft paper text with LLMs
