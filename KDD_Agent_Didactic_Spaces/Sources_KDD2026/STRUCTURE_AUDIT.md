# ADS Paper Structure Audit: Main vs Supplementary

## Core Claims (what the paper MUST prove)

| # | Claim | Key Evidence | Status |
|---|-------|--------------|--------|
| C1 | Pareto >> scalarization | 158 vs 15 options (10.5x) | OK in main |
| C2 | All 7 universities contribute uniquely | Pairwise Jaccard = 0 | OK but buried |
| C3 | Tau provides governance control | Tau sweep 25→175 | OK |
| C4 | Practitioners value transparency | 4.85/5 rating | OK |
| C5 | Cross-national specialization | KTH leads via learner (+17.8%) | KEY insight, well presented |

---

## MAIN PAPER: What MUST Stay

### Essential Figures
- **Fig 1 (Architecture)** - Shows the paradigm shift, keep
- **F3 (Pareto vs Scalarization bar)** - Proves C1, keep
- **F1 (Market-mission scatter)** - Shows trade-off diversity, keep
- **Fig 3 (Practitioner ratings)** - Proves C4, keep

### Essential Tables
- **T0 (Dataset overview)** - Establishes benchmark, keep
- **T2 (Tau sweep)** - Proves C3, keep
- **T3 (Encoder ablation)** - Robustness, keep
- **T_market_alignment** - Supports C5, keep
- **T_objective_specialization** - KEY for C5, keep
- **T9 (Pilot results)** - Supports C4, keep

---

## TO SUPPLEMENTARY: What to Move Down

### 1. Introduction - TRIM
**Problem:** Same idea repeated twice (paragraphs 3-4)
```
Current: "ADS/MAPE is a practical operationalization..."
Then again: "We therefore argue that pathway evaluation..."
```
**Action:** Merge into one paragraph. Save ~100 words.

### 2. Paper Structure Paragraph - REMOVE
```latex
\paragraph{Paper structure.}
Section~\ref{sec:related} reviews adjacent work...
```
**Reason:** Just navigation, wastes space. Readers can see the sections.

### 3. Related Work - CONDENSE
**Current:** 4 subsections, each with "How ADS differs"
**Problem:** Repetitive structure, defensive tone
**Action:**
- Merge into 2 paragraphs: (a) Educational AI & Recommenders, (b) Multi-objective & Contestable AI
- Keep citations, cut "How ADS differs" to one sentence at end
- **Saves ~150 words**

### 4. Cross-stakeholder Calibration - TO SUPP
```latex
\paragraph{Cross-stakeholder calibration.}
Raw cosine similarities are not directly comparable...
```
**Reason:** Technical detail. Main paper: just say "z-score normalized for comparability"
**Saves ~80 words**

### 5. Directed Constraint Variant - TO SUPP
```latex
For governance scenarios where a specific stakeholder $a^\star$...
g_{a^\star}(x) = s_{a^\star}(x) - \max_{b\in\mathcal{A}\setminus\{a^\star\}} s_b(x).
```
**Reason:** Refinement, not core. Keep symmetric g(x), move directed to supp.
**Saves ~60 words**

### 6. Practitioner Quotes - KEEP ONE, MOVE REST
**Current:** P02, P07, P13, P06 quotes
**Action:** Keep best one (P02: "making these trade-offs unconsciously"), move rest to supp
**Saves ~80 words**

### 7. "Implications and Next Steps" Subsection - TO SUPP
**Reason:** Future work, not contribution.
**Saves ~120 words**

### 8. "Why we do not compare to SOTA" - CONDENSE
**Current:** Full paragraph explaining why no Course2Vec comparison
**Action:** One sentence: "We evaluate governance, not prediction, so recommender baselines are orthogonal."
**Saves ~80 words**

### 9. "Strict Provenance Mode" Paragraph - TO SUPP
**Reason:** Implementation detail for reproducibility appendix
**Saves ~40 words**

### 10. Detailed Robustness Numbers - SUMMARIZE
**Current:** "Jaccard ≈ 0.06 at ε=0.02, bootstrap preserves 82%, pairwise Jaccard=0..."
**Action:** Keep headlines, move details to supp table
**Saves ~60 words**

### 11. Failure Modes Paragraph - CONDENSE
**Current:** 4 failure modes + mitigation explanation
**Action:** Bullet list, half the words
**Saves ~80 words**

---

## FROM SUPPLEMENTARY: What to PULL UP

### 1. Edge Case Examples (currently "in Supplementary")
**Current:** F1 caption says "edge cases in Supplementary"
**Action:** Add 2-3 concrete course names to main paper
**Why:** Makes the claim tangible. "MIT 6.001 vs Stanford CS229" is more memorable than "max-market option"

### 2. Pareto Contribution by University Table
**Current:** Only in paper_constants as macros (ParetoKTH=52, etc.)
**Action:** Add inline or small table showing all 7 universities' contributions
**Why:** Directly proves C2 (all contribute)

### 3. Goal Sensitivity Finding
**Current:** In paper_constants but not in text (GoalCareerPareto=142, etc.)
**Action:** Add one sentence: "Different learner personas yield distinct frontiers (Jaccard 0.38-0.58)"
**Why:** Strengthens multi-stakeholder claim

---

## STRUCTURAL ISSUES

### Issue 1: Introduction Repetition
**Lines 130-134:** Two paragraphs saying the same thing
**Fix:** Merge, keep the crisper version

### Issue 2: Contribution List Too Long
**Current:** 5 bullet points
**Problem:** Items 4-5 are really one point (evaluation + validation)
**Fix:** Merge to 4 items max

### Issue 3: Problem Setting Section Too Short
**Current:** 3 one-liner paragraphs
**Fix:** Either expand with motivation OR merge into Methods

### Issue 4: Results Lack "So What"
**Example:** "MiniLM yields 158, MPNet yields 62"
**Missing:** What does this mean for practitioners?
**Fix:** Add interpretation: "Encoder choice matters—teams should validate on their domain"

### Issue 5: Conclusion Too Thin
**Current:** 2 sentences
**Fix:** Add: (1) main takeaway, (2) practical implication, (3) limitation acknowledgment

---

## PROPOSED SECTION STRUCTURE (v2)

```
1. Introduction (1 page)
   - Problem: scalarization hides trade-offs
   - Solution: ADS/MAPE
   - Contribution (4 bullets max)
   - [NO paper structure paragraph]

2. Related Work (0.5 page)
   - Educational AI & Skill Matching
   - Multi-objective & Contestable AI
   - [One "How ADS differs" at end]

3. ADS/MAPE Framework (1.5 pages)
   - 3.1 Problem Setting (merged)
   - 3.2 Shared Space & Lenses
   - 3.3 Feasibility Constraint
   - 3.4 Pareto Set & Explanations
   - Algorithm box

4. Practitioner Validation (0.75 page)
   - Formative interviews (condensed)
   - Pilot study (one quote)
   - [Implications to supp]

5. Experiments (2 pages)
   - 5.1 Dataset
   - 5.2 Main Results (Pareto vs baselines, Tau, Encoder)
   - 5.3 Cross-National Analysis (KEY finding)
   - [Robustness details to supp]

6. Limitations & Ethics (0.5 page)
   - [Failure modes condensed]

7. Conclusion (0.25 page)
   - Expanded with takeaway
```

**Target: 8 pages exactly**

---

## ACTION ITEMS

1. [ ] Merge Introduction paragraphs 3-4
2. [ ] Remove "Paper structure" paragraph
3. [ ] Condense Related Work to 2 subsections
4. [ ] Move calibration details to supp
5. [ ] Move directed constraint to supp
6. [ ] Keep only P02 quote, move rest to supp
7. [ ] Move "Implications and Next Steps" to supp
8. [ ] Condense "Why no SOTA comparison"
9. [ ] Move "Strict provenance" to supp
10. [ ] Summarize robustness, details to supp
11. [ ] Condense failure modes
12. [ ] Add concrete edge case names
13. [ ] Add university contribution breakdown
14. [ ] Expand conclusion

---

## WORD BUDGET

| Section | Current | Target | Action |
|---------|---------|--------|--------|
| Abstract | 180 | 180 | Keep |
| Introduction | 650 | 500 | Trim repetition |
| Related Work | 500 | 350 | Merge subsections |
| Problem/Method | 900 | 800 | Move details down |
| Stakeholder | 550 | 350 | Move implications, quotes |
| Experiments | 850 | 900 | Add interpretation |
| Limitations | 400 | 300 | Condense |
| Conclusion | 50 | 150 | Expand |
| **Total** | **4080** | **3530** | **-550 words** |

This creates room for:
- Concrete examples (+100)
- University breakdown (+50)
- Result interpretation (+100)
- Stronger conclusion (+100)
