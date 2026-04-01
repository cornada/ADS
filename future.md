# ADS Evolution Journal (future.md)

> Cumulative append-only log. NEVER overwrite previous entries.
> Each session: what was done, insights, failures, forks/opportunities.

---

## Session 2026-03-30 — Strategic Review + Project State Audit

### Context
- Colleague provided deep strategic analysis (TRIZ-style): formulation, contradictions, failure modes, positioning, morphological analysis, wave analysis
- Session goal: assess current state, verify deadlines, determine next actions

### What was done
1. **Deadline audit** — checked all 8 venues (KDD, IJCAI, AAMAS, ICAPS, AAAI, EDM, L@S, LAK)
   - ALL main-track deadlines have passed
   - KDD 2026 Cycle 2: submitted Feb 8, rebuttal Apr 4-17
   - IJCAI 2026: submitted Jan 19, author response Apr 7-10
   - EDM 2026: poster/demo deadline Apr 6-9 (still open)
   - L@S 2026: demo deadline Apr 14 (still open)

2. **Project state audit**
   - KDD paper: "Agent-Didactic Spaces: A Cross-National Educational Benchmark for Multi-Objective Pathway Evaluation"
   - Dataset: ~169K records, 27,555 courses, 7 universities, 3 countries
   - 189 experiment runs completed
   - Current branch: `kt20/outcome-sanity-curriculum-text`
   - No future.md existed prior to this session

3. **Colleague analysis synthesis**
   - Core reframe: "curriculum engineering" > "curriculum optimization"
   - Killer reviewer question: "NSGA-II is 20 years old. What's novel?" 
   - Answer: (a) domain formalization of didactic space + (b) real university data with regulatory constraints
   - EDM = right audience > prestigious audience (reviewers understand FGOS, curriculum pain)
   - TC2 (many venues = 0 submissions) — same MathY/LeNEPA failure pattern
   - Wave analysis: curriculum optimization = pre-birth on S-curve, publish anywhere for priority

### Insights
- **Deadline reality vs strategy mismatch**: Colleague's analysis assumed we're still choosing venue. If KDD/IJCAI already submitted, the game is rebuttal prep, not venue selection. Same arguments, different deliverable (1-page response vs full paper).
- **EDM poster/demo (Apr 6-9) = live hedge**: Even if KDD/IJCAI reject, an EDM poster establishes priority and puts work in front of the RIGHT audience.
- **189 experiment runs but no future.md**: Significant work happened without evolution logging. Pattern risk: work exists but context/decisions/failures are not captured.
- **Scope/ directory has 6 venue-specific drafts**: Confirms TC2 diagnosis — energy spread across IJCAI, AAMAS (x2), ICAPS, AAAI, KDD. All deadlines now passed.

### Mistakes and failures
- **No future.md until session 189+**: Lost all decision context from prior sessions. Why certain experiments were run, what failed, what was abandoned. This is the most expensive kind of technical debt — not code debt, but reasoning debt.
- **6 venue scope docs created but (likely) only 1-2 submitted**: Classic analysis paralysis pattern. The scope docs themselves are not wasted (reusable framing), but the time cost of maintaining 6 parallel framings was real.

### Forks and opportunities
1. **OPEN NOW: EDM 2026 poster/demo (Apr 6-9)** — 7 days. Can repackage existing KDD material. Right audience for curriculum optimization. Low effort if demo already exists (SPA UI).
2. **OPEN NOW: L@S 2026 demo (Apr 14)** — 15 days. Learning at Scale = similar audience to EDM.
3. **CRITICAL PATH: KDD rebuttal (Apr 4-17)** — Need to verify if submitted. If yes, prepare novelty defense.
4. **CRITICAL PATH: IJCAI author response (Apr 7-10)** — Need to verify if submitted. If yes, prepare response.
5. **FUTURE: AAAI 2027** — Deadlines likely Jul-Aug 2026. Enough time to iterate on any KDD/IJCAI feedback.
6. **FUTURE: EDM 2027, LAK 2027** — Domain venues. If top-venue strategy fails, domain venue = higher acceptance + right reviewers.
7. **FUTURE: Journal route** — Computers & Education, Interactive Learning Environments, IEEE TLT. No deadline pressure. Tool/system paper format.
8. **arXiv preprint** — Establish priority NOW regardless of venue outcome. Zero cost if paper already written.

### Open questions (need user confirmation)
- [ ] Was KDD Cycle 2 actually submitted?
- [ ] Was IJCAI actually submitted?
- [ ] Which of the 6 Scope/ drafts were submitted vs abandoned?
- [ ] Is the SPA demo deployable for EDM poster/demo submission?

---

## Session 2026-03-31 — ACM MM Dataset Track Submission (ADS-MM)

### Context
- ACM MM 2026 Dataset Track deadline: April 1 AoE (~20 hours from session start)
- KDD paper exists (169K records, 7 unis), but text-only = weak MM fit
- Colleague's v0.4 analysis: conditional GO, run API test first

### What was done
1. **MIT OCW API test** — user manually verified 78% video density on 9 hand-picked courses → GO decision
2. **Data collection script** (`scripts/fetch_ocw_multimodal.py`)
   - Parsed MIT OCW sitemap.xml → 2,569 course slugs
   - Mapped slugs to course numbers via regex (fixed special topic courses: S-prefix, A-prefix, ES-prefix)
   - Fetched data.json for 1,689 courses, 0 errors
   - Matched 989 to ADS dataset (58.6%)
   - Output: `data/multimodal/ocw_video_metadata.jsonl`
3. **Multimodal statistics** (real numbers, post-collection):
   - Video (any type): 260 (15.4%) — NOT 78% from hand-picked test
   - Audio: 34 (2.0%), Visual: 196 (11.6%)
   - Any multimedia: 396 (23.4%)
   - Notes: 1,100 (65.1%), Assessment: 843 (49.9%)
   - Mean richness: 1.45 ± 0.98
4. **Cross-modal baselines** (`experiments/scripts/crossmodal_baselines.py`):
   - Task 1 (richness prediction): TF-IDF ridge MAE=0.501, **Spearman rho=0.612** (42% above dept prior)
   - Task 2 (classification): macro_F1=0.196 (=majority, class imbalance issue)
5. **Paper** (`ACM_MM_Dataset/Sources/main.tex`):
   - 5 pages, ACM sigconf, single-blind, clean compile
   - 3 figures (resource distribution, richness histogram, co-occurrence matrix)
   - 3 tables (dataset comparison, dataset composition, baselines)
   - 15 references (verified, no fabricated refs)
6. **Figure generation** (`experiments/scripts/generate_mm_figures.py`)

### Insights
- **78% vs 15.4% video density** — hand-picked CS courses massively overestimate OCW-wide video prevalence. Dept 6 (EECS) and 18 (Math) are outliers. Most OCW courses (Civil, Biology, etc.) are notes+assessment only. Lesson: ALWAYS do population-level test, not convenience sample.
- **Richness prediction works (rho=0.612)** — course TEXT genuinely predicts multimedia STRUCTURE. This is a real cross-modal finding: descriptions encode "instructional investment" signal.
- **Classification fails on minority labels** — 15% video base rate + simple logistic regression = all-negative prediction. Need class weighting or better models. This is expected and actually useful for paper: "establishes baseline, motivates future work."
- **Slug-to-course mapping** — OCW uses `dept-number-title-semester` slugs. Special topics (S-prefix), experimental (A-prefix), joint (J-suffix) courses needed regex expansion. Initial mapping: 1,624/2,569; after fix: 1,689/2,569 (880 unmapped = non-standard formats or pages without course data).

### Mistakes and failures
- **Initial VIDEO_TYPES too narrow** — only matched "Lecture Videos" (211), missed "Demonstration Videos" (21), "Tutorial Videos" (11), "Simulation Videos" (11), etc. Expanded set captures 260. Should have checked actual field values BEFORE defining constants.
- **Go/no-go threshold miscalibrated** — set at 50% video density, tested on biased sample. If tested on random sample first, would have gotten ~15% and might have chosen different strategy. Decision still correct (rho=0.612 validates the approach), but the reasoning was fragile.
- **Classification baseline too simple** — hand-rolled logistic regression (100 epochs, lr=0.1) doesn't converge for imbalanced labels. Should have used class weighting from the start. Not critical for paper (simple baseline is fine for dataset track), but wasted 10 minutes debugging.

### Future ideas to check
- [x] **SBERT embeddings baseline** — DONE: rho=0.633 (+0.021 vs TF-IDF), macro F1=0.572 (2.9x improvement)
- [ ] **Per-department video prediction** — EECS dept probably has F1>0.5, humanities near 0. Stratified analysis.
- [ ] **Extend multimodal to other universities** — UC Berkeley Webcast Archive (Internet Archive) is the top priority
- [x] **Cross-modal embedding space** — DONE: PC1↔richness r=0.41, video centroid shift=0.17
- [ ] **This paper as EDM/L@S submission** — if ACM MM rejects, repackage for EDM (Apr 6-9) or L@S (Apr 14)

---

## Session 2026-03-31 (continued) — Full-Scale Experiments + Paper Finalization

### What was done (continuation)
4. **MISIS_data integration** — discovered 11,323 parsed RPDs at MISIS_data/parser/output/full_dataset.json
   - Added MISIS as 8th university (Russia, 4th country): 2,694 unique courses
   - Hours breakdown, 39 FGOS competency codes, 99,676 prerequisite links
   - Title-based join to canonical dataset: 2,061/2,175 matched (95%)

5. **9 data islands problem discovered and solved**
   - Hash-based IDs differ across unified_v3/v4/v5/multimodal — 0 overlap for same courses
   - Built canonical ID system: `{institution}:{catalog_id}`
   - Built `scripts/build_canonical_dataset.py` — 36,192 unified records with 7 layers
   - Canonical dataset: `data/canonical/ads_mm_canonical.jsonl`

6. **Full-scale SBERT latent space analysis** (36,192 records, all-MiniLM-L6-v2, 384d)
   - Took ~7 hours on CPU (thermal throttling). RunPod available but not used.
   - Unified didactic space CONFIRMED: separation ratio 0.70
   - PC1↔richness r=0.41 (IDENTICAL to 10K sample — rock solid)
   - 36% cross-national nearest neighbors
   - MISIS tightest cluster (0.54), MIT widest (0.90) — FGOS vs decentralized
   - Course-Mission distance (0.40) < Course-Job (0.60) — missions encoded in curricula

7. **16 comprehensive figures generated**
   - Institution distance heatmap, intra-distance bars, cross-national NN, type distances,
     course-job alignment, PC-richness correlation, separation ratio gauge, MM resource types,
     richness by dept, MISIS hours breakdown, competency co-occurrence, prereq stats,
     dataset composition, baseline comparison, layer coverage heatmap, key findings dashboard

8. **Paper critical review and revision**
   - Added latent space section (strongest content, was NOT in paper before review)
   - Added MOOCCubeX (230K videos), Course-Skill Atlas (3M syllabi), M3AV to related work
   - Converted all bullet lists to prose (anti-LLM-slop)
   - Added Discussion section with three substantive interpretations
   - Added Didactic Space formalism (2 equations from IJCAI draft)
   - Expanded refs from 15 → 50 (added ACM MM citations, CLIP, BERT, etc.)
   - Expanded all acronyms at first mention
   - Fixed richness scale inconsistency (0-6 everywhere)
   - Removed Task 3 from contributions (was defined but not evaluated)
   - Final: 7 pages (6 content + 1 refs), 50 refs, 2 equations, 7 figures, 3 tables

9. **Multi-agent research sweep** (4 parallel agents)
   - Novelty: MODERATE (MOOCCubeX has richer content, Course-Skill Atlas has more scale)
   - MM fit: WEAK→MODERATE after strengthening (images + SBERT + latent space)
   - Missed sources: UC Berkeley Webcast Archive (#1 priority), Yale OCW, PhET simulations
   - Latent space beyond university: "blue ocean" — no one does education→life outcomes

10. **Release package + Zenodo upload**
    - DOI: 10.5281/zenodo.19363580
    - Archive: release/ads-mm-complete.tar.gz (124 MB)
    - HOW_TO_WRITE_PAPER.md operational playbook

### Insights
- **Unified didactic space is REAL** — not artifact of sampling. Stable across 10K and 36K. Courses from MIT, KTH, MISIS mix in latent space despite language/tradition/regulation differences.
- **PC1↔richness r=0.41 is the strongest finding** — text-only embeddings capture multimedia structure as geometric property. This is publishable independently of the dataset.
- **MISIS is the most valuable data addition** — not because of size (2,694 courses) but because of DEPTH: prerequisite graph (99K links), competency codes (39 FGOS), hours breakdown, 5-year longitudinal. No other university in the dataset has this richness.
- **Anti-LLM-slop matters** — user immediately spotted consecutive bullet lists. Community papers use dense prose. This is a real reviewer signal.
- **Data consistency is technical debt** — 9 data islands with incompatible hashes accumulated over months. Should have designed canonical IDs from day 1.
- **Venue style research BEFORE writing** — downloading 7 accepted papers from MM 2024-2025 revealed: 45 refs median, prose-heavy, no thin sections. Our first draft had 15 refs and bullet-list-heavy layout.

### Mistakes and failures
- **Didn't add latent space findings to paper initially** — the strongest content (ratio 0.70, PC1↔r=0.41) was produced but NOT included in the paper. Only discovered during self-review. Always check: "is my best result in the paper?"
- **Full-scale SBERT took 7 hours on CPU** — should have used RunPod GPU from start (3 min). Sunk cost of waiting. For next time: if >1 hour on CPU, switch to GPU.
- **9 data islands accumulated silently** — each ingest pipeline hashed IDs differently. No integration tests caught this. Need CI check: "can all datasets be joined?"
- **MISIS join yielded 33/2175 initially** — canonical ID stripped hash suffix that MISIS needs (direction_code = program, not course). Debugging cost 20 minutes. Lesson: always check join count immediately.
- **Added fabricated arXiv ref "EduCLIP"** — LLM hallucination. Caught during reference audit. Rule: verify EVERY new reference against DBLP/Scholar.

### Status after session
- **Submitted**: ACM MM 2026 Dataset Track (Apr 1 AoE)
- **Upcoming**: KDD rebuttal (Apr 4-17), IJCAI response (Apr 7-10), EDM poster (Apr 6-9)
- **Created**: HOW_TO_WRITE_PAPER.md playbook
- **Memory**: Updated with 6 entries (strategy, workflow, paper rules, data arch, latent space, MISIS)

### Open questions for next session
- [ ] Was KDD Cycle 2 actually submitted? (Rebuttal opens Apr 4)
- [ ] Was IJCAI actually submitted? (Author response Apr 7-10)
- [ ] Should we submit EDM poster/demo? (Deadline Apr 6-9)
- [ ] Run latent space analysis on RunPod GPU with multiple encoders (MPNet, BGE)
- [ ] UC Berkeley Webcast Archive — would add video modality for 2nd institution
- [ ] "Latent life space" concept paper — blue ocean, no competitors
