# ADS Project Makefile
# Reproducibility commands for IJCAI paper experiments

.PHONY: help install test lint smoke reproduce_small reproduce_full paper clean dev exp sweep

# Default target
help:
	@echo "ADS Project - Available targets:"
	@echo ""
	@echo "  install          Install dependencies"
	@echo "  install-embed    Install with embedding support (sentence-transformers)"
	@echo "  test             Run all tests"
	@echo "  lint             Run linting checks"
	@echo "  smoke            Run smoke test (toy dataset)"
	@echo ""
	@echo "  reproduce_small  Run minimal reproducibility test (stub embeddings only)"
	@echo "  reproduce_full   Run full experiment matrix (requires sentence-transformers)"
	@echo "  paper            Generate paper artifacts from existing results"
	@echo ""
	@echo "  dev              Start API server (alias for api)"
	@echo "  api              Start API server"
	@echo "  clean            Remove generated artifacts"

# Installation
install:
	pip install -e ".[dev,exp]"

install-embed:
	pip install -e ".[dev,exp,embed]"

# Testing
test:
	pytest -q

lint:
	ruff check .

# Development
dev:
	uvicorn ads_api.main:app --reload

api:
	uvicorn ads_api.main:app --reload

# Smoke test
smoke:
	python -m experiments.run dataset=toy embedding=stub lenses=identity outputs.run_id=smoke_test

exp:
	python -m experiments.run dataset=toy embedding=stub lenses=identity

# Reproducibility targets
reproduce_small:
	@echo "========================================"
	@echo "Running minimal reproducibility test"
	@echo "========================================"
	@echo ""
	@echo "Step 1: Running tests..."
	pytest -q
	@echo ""
	@echo "Step 2: Running experiment matrix (stub only)..."
	python -m experiments.paper_experiments --quick
	@echo ""
	@echo "Step 3: Verifying outputs..."
	@test -f experiments/reports/paper/paper_tables/full_results.csv || (echo "ERROR: Missing full_results.csv" && exit 1)
	@test -f experiments/reports/paper/figures/fig_A1_lens_ablation.png || (echo "ERROR: Missing lens ablation figure" && exit 1)
	@echo ""
	@echo "SUCCESS: Minimal reproducibility test passed"
	@echo "See experiments/reports/paper/ for outputs"

reproduce_full:
	@echo "========================================"
	@echo "Running full reproducibility suite"
	@echo "========================================"
	@echo ""
	@echo "Step 1: Running tests..."
	pytest -q
	@echo ""
	@echo "Step 2: Running full experiment matrix..."
	python -m experiments.paper_experiments
	@echo ""
	@echo "Step 3: Verifying outputs..."
	@test -f experiments/reports/paper/paper_tables/full_results.csv || (echo "ERROR: Missing full_results.csv" && exit 1)
	@test -f experiments/reports/paper/figures/fig_A1_lens_ablation.png || (echo "ERROR: Missing lens ablation figure" && exit 1)
	@test -f experiments/reports/paper/summary.json || (echo "ERROR: Missing summary.json" && exit 1)
	@echo ""
	@echo "SUCCESS: Full reproducibility suite passed"
	@echo "See experiments/reports/paper/ for outputs"

# Paper artifact generation (from existing results)
paper:
	python -m experiments.paper_experiments --skip-run

# Hydra multirun sweeps
sweep:
	python -m experiments.run -m dataset=toy embedding=stub,sbert_allminilm lenses=identity,diagonal,learned

sweep-lenses:
	python -m experiments.run -m dataset=toy embedding=stub lenses=identity,diagonal,learned outputs.run_id=sweep_lens

sweep-embeddings:
	python -m experiments.run -m dataset=toy embedding=stub,sbert_allminilm lenses=identity outputs.run_id=sweep_emb

# Dataset verification
verify-datasets:
	@echo "Verifying dataset integrity..."
	@python -c "from ads_core.ingest.toy_dataset import build_toy_artifacts; arts = build_toy_artifacts(); print(f'Toy dataset: {len(arts)} artifacts')"

# Cleanup
clean:
	rm -rf experiments/reports/paper_*
	rm -rf experiments/reports/smoke_*
	rm -rf experiments/reports/kt*
	rm -rf data/api_runs
	rm -rf data/telemetry
	rm -rf .pytest_cache
	rm -rf **/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
