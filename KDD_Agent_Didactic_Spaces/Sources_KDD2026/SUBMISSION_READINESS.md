# ADS Paper Submission Readiness Assessment
## KDD 2026 Datasets & Benchmarks Track

**Date:** 2026-02-05
**Current Status:** 9 pages, 37 citations, 9 figures, 6 tables

---

## Overall Score: **7.2 / 10** — Almost Ready

| Category | Score | Status |
|----------|-------|--------|
| Template & Format | 9/10 | ✅ Ready |
| Core Claims | 8/10 | ✅ Ready |
| Dataset Quality | 7/10 | ⚠️ Minor gaps |
| Experiments | 8/10 | ✅ Ready |
| Reproducibility | 6/10 | ⚠️ Needs work |
| Writing Quality | 7/10 | ⚠️ Minor polish |
| D&B Track Fit | 8/10 | ✅ Good fit |

---

## 1. Template & Format (9/10) ✅

| Requirement | Status | Notes |
|-------------|--------|-------|
| ACM sigconf template | ✅ | `\documentclass[sigconf,review]{acmart}` |
| Page limit (8 content) | ⚠️ | Currently 9 pages — need to cut ~1 page |
| Single-blind | ✅ | Author names can be visible |
| Double-column | ✅ | ACM format |
| CCS concepts | ✅ | 3 concepts defined |
| Keywords | ✅ | 5 keywords |
| References format | ✅ | ACM-Reference-Format.bst |

**Action needed:** Cut ~1 page of content to meet 8-page limit.

---

## 2. Core Claims (8/10) ✅

| Claim | Evidence | Strength |
|-------|----------|----------|
| Pareto >> scalarization | 158 vs 15 (10.5×) | ✅ Strong |
| Cross-university diversity | 7/7 contribute, Jaccard=0 | ✅ Strong |
| τ governance control | Tau sweep 25→175 | ✅ Strong |
| Practitioner value | 4.85/5 transparency | ✅ Strong |
| Cross-national specialization | +17.8% learner gap | ✅ Strong |
| Encoder robustness | Seed Jaccard=1.0 | ✅ Strong |
| Market drift stability | 3% turnover | ✅ Strong |

**Missing claims that could strengthen:**
- [ ] Comparison with non-Pareto multi-objective methods (e.g., weighted Chebyshev)
- [ ] User study on decision quality (not just perceived usefulness)

---

## 3. Dataset Quality (7/10) ⚠️

### What's Good
| Aspect | Status |
|--------|--------|
| Scale | ✅ 32,728 courses, 7 universities, 3 countries |
| Diversity | ✅ US + Europe, multiple disciplines |
| Public sources | ✅ OCW, O*NET, KOPPS, DRPS |
| Provenance | ✅ Manifest with hashes |

### What's Missing for D&B Track

| Gap | Priority | Effort |
|-----|----------|--------|
| **Dataset DOI** | 🔴 High | Upload to Zenodo |
| **Croissant metadata** | 🔴 High | Create JSON-LD file |
| **Datasheet for Datasets** | 🟡 Medium | Write Gebru et al. template |
| **Benchmark tasks defined** | 🟡 Medium | Specify input/output format |
| **Baseline results file** | 🟡 Medium | CSV with reproducible numbers |
| **License clarity** | 🟡 Medium | Explicit per-source licenses |

**Action needed:**
1. Upload dataset to Zenodo → get DOI
2. Create `croissant.json` metadata
3. Add Datasheet section to Supplement

---

## 4. Experiments (8/10) ✅

| Experiment | Completeness | Visualization |
|------------|--------------|---------------|
| Main Pareto evaluation | ✅ Complete | F3 bar chart |
| Tau sweep | ✅ Complete | F2 curve + T2 table |
| Encoder ablation | ✅ Complete | T3 table |
| Seed stability | ✅ Complete | Jaccard=1.0 |
| Cross-university | ✅ Complete | F5 heatmap |
| Market drift | ✅ Complete | Numbers in text |
| Goal sensitivity | ⚠️ Mentioned in constants but not in paper | Missing |
| Practitioner study | ✅ Complete | Fig 3 + T9 |

**Minor gaps:**
- Goal sensitivity analysis results are in `paper_constants.tex` but not presented in paper
- Lens ablation mentioned but not detailed

---

## 5. Reproducibility (6/10) ⚠️

| Element | Status | Location |
|---------|--------|----------|
| Random seeds | ⚠️ Mentioned, not specified | Should be seed=0,1,2 |
| Config files | ⚠️ Exist but not referenced | `experiments/conf/` |
| Run commands | ❌ Not provided | Need exact CLI |
| Hardware specs | ❌ Not provided | Add to Supplement |
| Runtime numbers | ✅ 18s mentioned | In text |
| Code availability | ⚠️ "Upon acceptance" | Need URL or DOI |
| Data availability | ⚠️ "Upon acceptance" | Need Zenodo DOI |

**Action needed:**
1. Add reproducibility appendix with exact commands
2. Specify seeds explicitly (0, 1, 2)
3. Add hardware specs
4. Create GitHub release or Zenodo code deposit

---

## 6. Writing Quality (7/10) ⚠️

### Strengths
- Clear problem statement
- Good use of figures
- Practitioner quotes add credibility
- Cross-national finding is compelling

### Issues to Fix

| Issue | Location | Fix |
|-------|----------|-----|
| Repetition in Intro | Lines 126-135 | Already fixed in v2 |
| "industry partner (anonymized)" | Throughout | Either name or remove |
| Some passive voice | Methods | Minor polish |
| Conclusion could be stronger | Last section | Add one-sentence takeaway |

### Missing Elements
- [ ] **Limitations of the dataset itself** (not just the method)
- [ ] **Comparison table** with related work (who does what)
- [ ] **One concrete example** showing a Pareto trade-off decision

---

## 7. D&B Track Fit (8/10) ✅

### Track Requirements Check

| Requirement | Status | Notes |
|-------------|--------|-------|
| Novel dataset | ✅ | 7 universities, 3 countries |
| Benchmark tasks | ⚠️ | Implicit — need explicit definition |
| Baseline methods | ✅ | Scalarization baselines |
| Evaluation metrics | ✅ | Jaccard, Hypervolume, Pareto size |
| Reproducibility | ⚠️ | Needs commands + DOI |
| Community value | ✅ | Multi-stakeholder EdTech gap |

### What Reviewers Will Look For

| Criterion | Our Answer | Strength |
|-----------|------------|----------|
| Is the dataset novel? | Yes — first multi-stakeholder curriculum benchmark | ✅ |
| Is it large enough? | 32K courses, 175K records | ✅ |
| Is it well-documented? | Partially — need Datasheet | ⚠️ |
| Are baselines fair? | Yes — scalarization is standard | ✅ |
| Is the task well-defined? | Mostly — need clearer I/O spec | ⚠️ |
| Can others reproduce? | Partially — need code/data DOI | ⚠️ |

---

## 8. Pre-Submission Checklist

### Must Do (Blocking)

- [ ] **Cut to 8 pages** — currently 9
- [ ] **Upload dataset to Zenodo** → get DOI
- [ ] **Add reproducibility section** with exact commands
- [ ] **Create Croissant metadata** (KDD 2026 requirement)
- [ ] **Update author block** (currently "Anonymous")
- [ ] **Verify all figures render at 300+ DPI**

### Should Do (Improves Score)

- [ ] Add Datasheet for Datasets (Supplement)
- [ ] Add one concrete Pareto decision example
- [ ] Add comparison table with related work
- [ ] Specify benchmark task format (input → output)
- [ ] Add goal sensitivity results to main paper
- [ ] Add hardware specs to Supplement

### Nice to Have

- [ ] Interactive demo URL
- [ ] Pre-registration of analysis
- [ ] Additional encoder (e.g., E5, BGE)

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Desk reject for format | Low | Fatal | Check page limit |
| Desk reject for no DOI | Medium | Fatal | Upload to Zenodo NOW |
| Weak reproducibility score | Medium | -1 to -2 points | Add commands + configs |
| "Incremental" criticism | Low | -1 point | Emphasize cross-national finding |
| "Limited generalization" | Medium | -1 point | Acknowledge in Limitations |
| Missing Croissant | Medium | -1 point | Create before submission |

---

## 10. Recommended Actions (Priority Order)

### This Week (Before Submission)

1. **Upload to Zenodo** (2 hours)
   - Package: courses.csv, artifacts.jsonl, manifest.json
   - Get DOI
   - Add DOI to paper

2. **Create Croissant metadata** (1 hour)
   - Use template from mlcommons
   - Include in Zenodo package

3. **Cut 1 page** (2 hours)
   - Move "Implications for Practice" to Supplement
   - Condense Related Work subsections
   - Remove one figure (F4 or F6)

4. **Add reproducibility commands** (1 hour)
   ```bash
   python -m experiments.run dataset=unified_v4 \
       embedding=sbert_minilm seed=0
   ```

5. **Update author block** (5 minutes)

### Before Camera-Ready

- Datasheet for Datasets
- Hardware specs
- Goal sensitivity in main paper
- Comparison table

---

## Summary

**Current State:** Strong paper with solid experimental evidence. Main gaps are administrative (DOI, Croissant, reproducibility commands) rather than scientific.

**Estimated Acceptance Probability:**
- If submitted as-is: **40-50%** (reproducibility/format concerns)
- With recommended fixes: **60-70%** (competitive for D&B track)

**Key Differentiator:** The cross-national finding (KTH leads Pareto via learner objective) is novel and surprising — emphasize this in rebuttal if needed.

**Bottom Line:** 2-3 hours of work to address blocking issues, then ready to submit.
