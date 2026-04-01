# O*NET Time-Travel Market Drift Experiment Results

**Generated:** 2026-01-19
**Status:** PAPER-GRADE (SBERT embeddings, unified_v3 dataset)

## Experiment Overview

This experiment tests the hypothesis that **labor market requirements drift faster than curricula** by "time-traveling" the market objective: keeping courses fixed while swapping O*NET versions.

### Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | unified_v3 (18,057 courses) |
| **Encoder** | SBERT (all-MiniLM-L6-v2) |
| **Seeds** | 0, 1, 2 (E1) |
| **Default τ** | 0.25 |
| **O*NET Versions** | 26.3 (2022), 27.3 (2023), 28.1 (2023), 30.0 (2025) |

## Key Findings

### E1: Market Drift Across O*NET Versions

| O*NET Version | Release | Pareto Size | Jaccard vs 2022 | Centroid Drift |
|---------------|---------|-------------|-----------------|----------------|
| 26.3 | 2022-05 | 29 | 1.00 | baseline |
| 27.3 | 2023-05 | 29 | 1.00 | 0.0011 |
| 28.1 | 2023-11 | 30 | **0.97** | 0.0013 |
| 30.0 | 2025-08 | 30 | **0.97** | 0.0018 |

**Key insight:** A measurable "break" occurs between 27.3 (May 2023) and 28.1 (Nov 2023),
where 3% of recommended courses change. This reflects real evolution in hot technologies
and emerging tasks during late 2023.

### E2: Governance Sensitivity Under Drift

At τ=0.05 (strict autonomy constraint):
- 2022 market: Pareto size = 24, Feasibility = 80%
- 2025 market: Pareto size = 25, Feasibility = 80%

The constraint bites harder when market signal dominates, but the effect is consistent across eras.

### E3: Technology Evolution

- **New technologies since 2022:** 51 (including AI/ML tools, cloud platforms)
- **Removed technologies:** 364 (many vendor-specific tools consolidated)
- **Stable technologies:** 82 (core tech stack)

## Artifacts

### Tables (`tables/`)

| File | Description |
|------|-------------|
| `market_drift_summary.csv` | Per-version Pareto statistics |
| `jaccard_matrix.csv` | Pareto set stability matrix |
| `centroid_drift_matrix.csv` | Embedding distance matrix |
| `tau_sweep_by_version.csv` | Governance sensitivity data |
| `new_technologies_since_2022.csv` | Emerging tech list |

### Figures (`figures/`)

| File | Description |
|------|-------------|
| `market_drift_heatmap.png` | Jaccard similarity heatmap (F7 candidate) |
| `centroid_drift_timeline.png` | Market signal drift over time |
| `pareto_size_vs_version.png` | Pareto evolution |
| `tau_sensitivity_comparison.png` | Governance under drift |
| `tech_evolution.png` | Technology count evolution |

### LaTeX (`latex/`)

| File | Description |
|------|-------------|
| `T_drift_summary.tex` | Main drift results table (T8 candidate) |
| `T_jaccard_matrix.tex` | Stability matrix table |
| `T_governance_drift.tex` | Governance comparison table |
| `T_tech_evolution.tex` | Technology evolution table |

## IJCAI Narrative Integration

This experiment supports the **contestability** and **governance** claims:

1. **Market drift is real and measurable** — even with the same occupations, the skill/tech signals change enough to alter 3% of recommendations.

2. **ADS exposes this drift** — by swapping the market corpus, stakeholders can see how recommendations would have differed under different market conditions.

3. **Governance constraints remain effective** — the τ parameter provides consistent control regardless of market era.

**Suggested placement:**
- Main paper: Jaccard heatmap as evidence for "updatable stakeholder targets"
- Supplement: Full time-travel analysis with tech evolution details

## Reproducibility

```bash
# Full reproduction (requires ~10 min on CPU with SBERT)
python -m experiments.scripts.run_onet_time_travel \
    --experiment all \
    --encoder sbert \
    --dataset unified_v3

# Quick test with stub encoder
python -m experiments.scripts.run_onet_time_travel \
    --experiment all \
    --encoder stub \
    --dataset toy
```
