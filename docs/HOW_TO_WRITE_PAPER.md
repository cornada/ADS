# How To Write a Conference Paper — Operational Playbook

Extracted from ADS-MM session (ACM MM 2026 Dataset Track, 31 March 2026).
Everything below is battle-tested: mistakes are documented because they cost time.

---

## Phase 0: Go / No-Go (30 min)

### Checklist before starting
- [ ] **Registration deadline** — many venues require abstract/title registration 5-7 days before paper deadline. Check FIRST.
- [ ] **Page limit** — know exact format: "6 pages + 2 refs" ≠ "8 pages total"
- [ ] **Review type** — single-blind (authors visible) vs double-blind (anonymize)
- [ ] **Dataset link required at submission?** — some tracks require live data access at review time
- [ ] **Dual submission rules** — different track ≠ dual submission, but different FRAMING required

### Go/No-Go decision
Run ONE experiment (15 min, $0) to validate feasibility before committing hours.
Example: "fetch 10 API records, check data density" — NOT "build full pipeline then check."

**Guardrail**: Hand-picked samples LIE. Our test showed 78% video density on 9 CS courses; population reality was 15.4%. ALWAYS test on random sample.

---

## Phase 1: Data (2-4 hours)

### Rules
1. **Script first, manual never.** Every data collection = reproducible Python script with `--max-courses N` flag for testing.
2. **Test on 10 records** before full run. Check: field names match expectations? Data density? Nulls?
3. **Polite scraping** — 0.3-0.5s delay between requests, User-Agent header.
4. **Save raw data outside /tmp.** /tmp is wiped on reboot. Always write to project directory.
5. **Provenance** — every record needs: source_url, timestamp, content_hash.

### Output
- `data/{name}/*.jsonl` — one record per line, UTF-8
- `data/{name}/collection_summary.json` — stats, timestamp, error count
- Script at `scripts/fetch_{name}.py` — reproducible, documented

### Guardrails
- Resource type names vary (e.g., "Lecture Videos" vs "Videos" vs "Lecture Video"). Check actual field values BEFORE defining constants.
- Match rates between datasets will be <100%. Report honestly. Don't inflate.

---

## Phase 2: Experiments (2-4 hours)

### Baseline strategy (minimum viable)
1. **Majority/random baseline** — establishes floor
2. **Simple feature baseline** (TF-IDF + linear model) — shows signal exists
3. **Pretrained embedding baseline** (SBERT + balanced classifier) — state of the art

### Rules
1. **Class imbalance** — if minority class <20%, MUST use `class_weight='balanced'` or equivalent. Simple logistic regression defaults to majority class. We lost 30 min debugging this.
2. **Report all metrics** — don't hide failures. If classification F1=0.196, report it and explain why.
3. **Fixed seed** — seed=42, document it. Reproducibility is non-negotiable.
4. **Save results as JSON** — not just console output. `experiments/reports/{name}/results.json`.

### What makes a finding publishable
- A number that surprises: "text predicts video availability at F1=0.658"
- A correlation that reveals structure: "PC1↔richness r=0.41"
- A negative result that teaches: "classification fails on minority labels without class weighting"

---

## Phase 3: Paper Writing (4-8 hours)

### Before writing: study the venue
1. Download 5-10 accepted papers from previous year
2. Note: section structure, figure types, reference count, writing style, tone
3. Match these conventions exactly — reviewers pattern-match unconsciously

### Structure (dataset track, 6+2 pages)

| Section | Pages | Content |
|---------|-------|---------|
| Introduction | 1-1.25 | Problem → gap → contributions (3-4 items) |
| Related Work | 0.75-1 | By topic, NOT by paper. Cite competitors honestly. |
| Formalism (if applicable) | 0.5 | Definitions, equations. Connects theory to experiments. |
| Dataset Description | 1.5 | Largest section. Collection process, statistics, composition. |
| Experiments | 1.5 | Statistics + baselines + analysis (latent space, etc.) |
| Discussion | 0.5 | What findings MEAN, not what numbers ARE |
| Limitations + Ethics | 0.25 | Honest, specific. Mention encoder sensitivity, annotation quality. |
| Conclusion | 0.25 | 3 specific findings + future directions. Not generic. |
| References | 1-2 | 40-50 refs for 6-page paper. Cite the venue. |

### Anti-LLM-slop guardrails

These are the most common tells that a paper was LLM-drafted. Avoid ALL of them:

| Tell | Fix |
|------|-----|
| **Bullet lists everywhere** | Convert to flowing prose. Max 1 itemize in entire paper (contributions). |
| **Two itemize blocks in a row** | NEVER. Always prose between lists. |
| **\paragraph{Bold Header.} for every idea** | Use prose flow. Max 2-3 \paragraph{} per section. |
| **Bold text mid-sentence for emphasis** | Remove. Use \emph{} sparingly, bold only in tables. |
| **"In this paper, we" repeated** | Say it ONCE in intro. Never again. |
| **"Furthermore", "Moreover", "Additionally"** | Delete. Start the sentence with the content. |
| **Trailing summary after each section** | Delete. Reader can read. |
| **"Comprehensive", "novel", "state-of-the-art"** | Remove unless backed by specific evidence. |
| **Generic conclusion** ("we release all data...") | Replace with 3 SPECIFIC findings. |
| **Numbered list of "contributions"** at start AND end | Once, in intro. Never repeat. |

### Abstract rules
- ~180 words. Lead with INSIGHT ("text predicts multimedia"), not counts ("30,249 courses").
- Structure: finding → dataset → scale → result → availability.
- Numbers required: dataset size, key metric, comparison.

### Reference rules
- Target 40-50 for 6-page paper
- Cite papers FROM the venue you're submitting to (3-5 minimum)
- Check arXiv preprints >1 year old — most have published versions by then. Use published version.
- Never fabricate references. Always verify against DBLP/Google Scholar.
- Competitors must be cited and compared honestly in a table. Don't pretend they don't exist.

### Acronym rules
- Expand EVERY acronym at first mention: "MIT OpenCourseWare (OCW)", "Principal Component Analysis (PCA)"
- After first mention, use acronym only
- Common exceptions that don't need expansion: URL, API, PDF, JSON

---

## Phase 4: Figures and Tables (1-2 hours)

### Rules
1. **Figure 1 = hero figure.** Pipeline diagram or visual showcase. This is what the reviewer sees first.
2. **Every figure must be referenced in text.** Unreferenced figures = wasted space.
3. **300 DPI minimum.** PDF format preferred for vector graphics.
4. **Consistent color palette** across all figures.
5. **Caption = self-contained.** Reader should understand the figure without reading body text.
6. **Tables: bold best results.** Standard convention.
7. **Comparison table is MANDATORY** for dataset papers. Show competitors honestly.

### Generate more figures than you need
Write a comprehensive figure generation script. Generate 15-20 figures. Select 5-7 best for paper. Keep rest for supplementary/rebuttal.

---

## Phase 5: Data Consistency Check (1 hour)

### The "9 islands" problem
Every data pipeline creates its own IDs. Before paper submission, verify:
- [ ] Can all data layers be JOINED? What's the join key?
- [ ] Are IDs stable across dataset versions?
- [ ] Do the numbers in the paper match the actual data files?
- [ ] Does the code reproduce the numbers in the paper?

### Canonical dataset
Build ONE file that joins all layers:
```python
# canonical.jsonl: {canonical_id, text, multimodal, activity, competencies, ...}
# Script: scripts/build_canonical_dataset.py
```

### Number consistency
- Grep for every `\newcommand` number in the tex file
- Verify each against the actual data source
- HeadlineRecords should match sum(actual records), not an old version

---

## Phase 6: Release Package (1 hour)

### Structure
```
release/{name}/
├── README.md          # Dataset card (HF-compatible)
├── LICENSE            # Explicit. CC BY-NC-SA 4.0 or similar.
├── data.jsonl         # Main dataset
├── images/            # If applicable
├── baselines/         # Experiment results (JSON)
└── scripts/           # Reproducible collection + evaluation
```

### Where to publish
1. **Zenodo** — DOI required for paper, versioned, ACM-compatible
2. **HuggingFace** — usability, `datasets.load_dataset()`, preview
3. **GitHub** — code only, not data

### Zenodo fields
- Resource type: Dataset
- License: match most restrictive component
- Keywords: 5-8, specific to venue
- Languages: all languages present in data

---

## Phase 7: Pre-submission Audit (30 min)

### Critical checklist
- [ ] Author info filled (not placeholder)
- [ ] DOI in paper matches uploaded dataset
- [ ] Paper compiles with 0 warnings, 0 undefined refs
- [ ] Page count within limit (content pages + ref pages separately)
- [ ] No TODO markers in text (`grep -i todo main.tex`)
- [ ] No red text or comments visible
- [ ] Dataset URL is live and accessible
- [ ] Supplementary materials deadline noted (often 1 week after paper)

### Anti-hallucination check
- [ ] Every cited reference exists (check 3 random ones on DBLP)
- [ ] Every number in abstract matches a number in body text
- [ ] Every claim has supporting evidence (table, figure, or citation)
- [ ] No "the first to..." claims without verifying competitors

---

## Mistakes Log (from this session)

| Mistake | Cost | Lesson |
|---------|------|--------|
| Hand-picked sample for go/no-go (78% vs 15.4%) | Near-fatal | ALWAYS random sample |
| VIDEO_TYPES too narrow (missed 49 courses) | 30 min | Check actual field values before defining constants |
| No class_weight='balanced' in classifier | 30 min | ALWAYS balance for minority class <30% |
| Hash-based IDs differ across dataset versions | 2 hours | Use catalog IDs (institution:number), not content hashes |
| 15 references for 6-page paper | Review risk | Target 40-50, cite the venue |
| Two itemize blocks in a row | LLM tell | Convert to prose, max 1 list per paper |
| Task 3 defined but not evaluated | Review risk | Don't claim undelivered contributions |
| HeadlineRecords from old dataset version | Misleading | Verify every \newcommand against actual data |
| arXiv ref with "TODO: verify" in note field | Embarrassing | Verify EVERY reference. LLMs fabricate. |
| Fabricated "EduCLIP" reference | Near-fatal | ALWAYS check references exist before citing |

---

## Time Budget (realistic)

For adapting an existing paper to a new venue with new data:

| Phase | Hours | Notes |
|-------|-------|-------|
| Go/No-Go | 0.5 | API test, deadline check, registration check |
| Data collection | 2-3 | Script + full run + stats |
| Experiments | 2-3 | 3 baselines + latent space analysis |
| Paper writing | 4-6 | Adapt from existing, NOT from scratch |
| Figures | 1-2 | Generate 15+, select 5-7 |
| Consistency check | 1 | Numbers, joins, IDs |
| Release package | 1 | Archive + Zenodo + README |
| Style audit | 1-2 | Anti-LLM-slop, acronyms, refs, venue fit |
| Pre-submission | 0.5 | Final checklist |
| **Total** | **13-19** | **Plan 20 hours to account for surprises** |
