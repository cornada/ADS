# ADS Paper - KDD 2026 Datasets & Benchmarks Track Version

This folder contains the paper formatted for KDD 2026 D&B track submission.

## Template

Uses the official ACM `acmart` template with `sigconf` format:
```latex
\documentclass[sigconf,review]{acmart}
```

## KDD 2026 D&B Track Requirements

- **Content pages**: 8 max (current: 6 pages)
- **References & Appendix**: Unlimited (but reviewers not obligated to read beyond page 8)
- **Review type**: Single-blind (author names visible)
- **Format**: Double-column ACM proceedings format

Source: [KDD 2026 D&B Call for Papers](https://kdd2026.kdd.org/datasets-and-benchmarks-track-call-for-papers/)

## Building

Requires TeX Live with `acmart` package (standard installation includes it).

```bash
cd Sources_KDD2026
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use latexmk:
```bash
latexmk -pdf main.tex
```

## Structure

```
Sources_KDD2026/
├── main.tex              # Main paper (ACM sigconf format)
├── paper_constants.tex   # Shared constants (dataset counts, metrics)
├── references.bib        # Bibliography
├── figures/              # All figures (400 DPI, bold fonts)
│   ├── F1_scatter_market_mission_edge_cases.png
│   ├── F2_tau_sweep_curve.png
│   ├── F3_scalarization_vs_pareto_bar.png
│   ├── F4_stability_jaccard_vs_eps.png
│   ├── F5_cross_university_overlap.png
│   ├── F6_scaling_curve.png
│   ├── F7_objective_specialization_radar.png
│   └── fig1_architecture_comparison.pdf
├── tables/               # All LaTeX tables
│   ├── T0_dataset_overview.tex
│   ├── T2_tau_sweep.tex
│   ├── T3_encoder_ablation.tex
│   ├── T9_pilot_results.tex
│   ├── T_market_alignment.tex
│   └── T_objective_specialization.tex
└── README.md             # This file
```

## Before Submission

1. Update author information (replace "Anonymous Author(s)")
2. Update `\acmSubmissionID{kdd26-db-XXX}` with actual submission ID
3. For camera-ready: update DOI, ISBN, copyright info
4. Verify page count is within limit

## Version Info

- **Dataset**: unified_v4 (7 universities, 3 countries, 32,728 courses)
- **Date**: 2026-02-05
