# ADS Implementation Spec (handoff-ready)
**Project:** Agent‑Didactic Spaces (ADS) — multi‑agent, multi‑objective framework for educational pathway evaluation  
**Audience:** engineer/ML colleague implementing the prototype + experiment harness  
**Date:** 2026‑01‑14

> Goal: deliver an IJCAI‑grade prototype + reproducible experiments that operationalize:
> 1) shared embedding space over educational/market/mission artifacts,
> 2) stakeholder “lenses” (multi‑objective scoring),
> 3) Pareto/contestable recommendation of next steps or short pathways,
> 4) ethics/fairness/autonomy constraints + counterfactual “recourse”.

---

## 0. Scope & non‑goals

### In scope (MVP)
- Ingest **open data** corpora (MIT course catalog/OCW style; OpenAlex; O*NET/ESCO).
- Build **unified embedding space** `E ⊂ R^d` for multiple entity types.
- Implement **stakeholder lenses** `L_a(v)=W_a v` (MVP: diagonal or low‑rank).
- Compute **vector evaluation** `F(option)` and **Pareto set** for candidate options.
- Provide **explanations**:
  - feature/lens contributions (“why on this axis?”),
  - counterfactual/recourse (“what minimal change moves you to another Pareto point?”).
- Provide **frontend demo** (dashboard) that:
  - visualizes trade‑offs,
  - supports “contestability” (toggle constraints/weights),
  - logs interactions for user study.

### Explicit non‑goals (do later)
- Full academic advising system with real student records.
- Full curriculum planning across years with prerequisites solved optimally (we can do greedy/beam for MVP).
- Production security/SSO; any PII pipelines.

---

## 1. External code: what we reuse and *how* (integration policy)

### License gate (mandatory before any copy/paste)
1) If repo has **permissive license** (MIT/BSD/Apache):  
   ✅ allowed to vendor/submodule or use as dependency.
2) If repo is **GPL/AGPL**:  
   ⚠️ do **NOT** mix into core repo unless we want our repo to become GPL; keep isolated.
3) If repo has **no LICENSE file**:  
   ❌ treat as “reference only” (read it, re‑implement ideas).

> Create `THIRD_PARTY_NOTICES.md` + keep commit hashes for reproducibility.

### Recommended reuse map (practical)
- **HypoCompass** (frontend/backend scaffold): reuse as *UI skeleton* and API pattern.
- **RobustX**: reuse as *counterfactual/recourse module* (if installable; otherwise implement a minimal recourse baseline).
- **EduStudio** or **pyKT**: reuse as *KT/CD baseline* (objective “learning gain / mastery”).
- **FAiRDAS**: reuse as *fairness stability evaluation harness* (metrics + logging).
- **MONB**: reuse as *MORL/bandit baseline* for sequential next‑step selection.
- **Improved Fair Rank Aggregation**: reuse as *fairness‑constrained aggregation baseline*.

Reference‑only (ideas/patterns):
- CoderAgent / Collaborative‑Agents / LivePoem / EMS assistant: use for study design and agent prompts only; do not copy code unless license verified.

---

## 2. Architecture overview

### High‑level components
1) **Ingestion**  
   Collect and normalize texts + metadata into a canonical “artifact store”.
2) **Embedding service**  
   Encode texts into vectors (cache, versioned).
3) **Lens service**  
   Apply stakeholder lenses, compute distances/scores.
4) **Planner / Pareto engine**  
   Generate candidate options (courses/projects) or short pathways; compute Pareto sets.
5) **Ethics & fairness layer**  
   Compute autonomy‑drift/capture metrics; enforce constraints/alerts; fairness metrics.
6) **Explanation layer**  
   - lens decomposition,
   - counterfactual recourse (robust if possible).
7) **API + UI**  
   REST endpoints + interactive dashboard (contestability).
8) **Experiment harness**  
   Reproducible pipelines, seeds, configs, reports, ablations.

### Recommended deployment topology (dev)
- `docker compose`:
  - `api` (FastAPI) — core ADS logic
  - `ui` (React) — dashboard
  - `redis` (optional) — cache job queues
  - `minio` or local FS — artifact store

---

## 3. Repository layout (monorepo)

```
ads/
  README.md
  pyproject.toml                 # Poetry/uv; pinned deps
  docker-compose.yml
  .env.example
  THIRD_PARTY_NOTICES.md

  packages/
    ads_core/                     # pure python library (no web)
      ads_core/
        __init__.py
        config/
          default.yaml
        data/
          schemas.py              # Pydantic models
          storage.py              # local FS / S3 abstraction
        ingest/
          mit_ocw.py
          course_catalog.py
          openalex.py
          onet_esco.py
          normalize.py
        embed/
          encoder.py              # interface + implementations
          cache.py
        lenses/
          base.py                 # Lens interface
          diagonal.py             # MVP W_a as diagonal
          lowrank.py              # optional
          learn_lenses.py         # optional training
        eval/
          distances.py
          objectives.py           # market/mission/univ/learner/gain/etc
          pareto.py
          constraints.py          # ethics/fairness/autonomy constraints
          explain.py              # lens contributions
          recourse.py             # wrapper to RobustX or fallback baseline
        plan/
          candidates.py           # candidate generation
          pathway.py              # beam/greedy short planning
        utils/
          hashing.py
          logging.py
          seeds.py

    ads_api/                      # FastAPI app
      ads_api/
        main.py
        routes/
          ingest.py
          embed.py
          evaluate.py
          pareto.py
          explain.py
          recourse.py
          experiments.py
        deps.py                   # dependency injection
        auth.py                   # optional (dev only)
      tests/

    ads_ui/                       # React app (can start from HypoCompass structure)
      package.json
      src/
        components/
        pages/
        api/
        state/
        viz/

  experiments/
    configs/
      mit_ocw_demo.yaml
      berkeley_outcomes.yaml
      kt_benchmark.yaml
    scripts/
      run_ingest.py
      run_embed.py
      run_eval.py
      run_ablation.py
      build_report.py
    reports/
      paper_tables/
      figures/
    notebooks/
      00_data_sanity.ipynb
      01_pareto_demo.ipynb
      02_user_study_analysis.ipynb

  baselines/
    edu_kt/                       # wrapper to EduStudio or pyKT
    monb/                         # git submodule or pinned fork
    fair_rank_agg/                # git submodule or pinned fork
    fairdas/                      # git submodule or pinned fork
    robustx/                      # wrapper / instructions
```

**Rule:** `ads_core` must stay runnable without UI/API; experiments call it directly.

---

## 4. Canonical data model (Pydantic)

### Entities
- `Artifact`: generic item stored in the corpus
  - `artifact_id`, `type`, `source_url`, `source_hash`, `timestamp`, `text`, `metadata`
- `EmbeddingRecord`:
  - `artifact_id`, `model_id`, `vector`, `created_at`
- `Stakeholder`:
  - `stakeholder_id` (market/mission/university/learner/ethics)
  - `lens_id`
  - `target_spec` (how to build target set `T_a`)
- `Option`:
  - `option_id`, `artifact_id`, `kind` (course/project/module), `action_spec`
- `EvaluationVector`:
  - `option_id`, `objectives: dict[str, float]`, `feasible: bool`, `violations: dict`
- `Pathway`:
  - list of `option_id`s + aggregated objectives

### Suggested minimal schemas (copy into `schemas.py`)
- `ArtifactType`: `MISSION`, `COURSE`, `SYLLABUS`, `LAB`, `PAPER_ABSTRACT`, `JOB_ROLE`, `SKILL`, `LEARNER_PROFILE`.

---

## 5. Core algorithms (MVP definitions)

### 5.1 Embeddings
- `v = r(text)` where `r` is a single encoder for all entity types.
- Store vectors with `model_id` and hashing for reproducibility.

**Interfaces**
- `Encoder.encode(texts: list[str]) -> np.ndarray`
- `Encoder.model_id: str`

**Baseline encoders**
- `sentence-transformers` local models (BGE, etc.)
- (Optional) remote API models — only if reproducibility plan is explicit.

### 5.2 Lenses
MVP lens: `L_a(v) = W_a v`, with `W_a = diag(w_a)`.

Where do weights come from?
- `w_market`: derived from skill lexicon overlap (O*NET/ESCO) OR learned on market corpus.
- `w_mission`: derived from mission corpus keywords/topics OR learned.
- `w_university`: derived from course outcomes corpus OR learned.
- `w_learner`: from learner self‑statement / preferences.

**Interfaces**
- `Lens.transform(v: np.ndarray) -> np.ndarray`
- `Lens.explain(v) -> dict` (top contributing dims/tokens if available)

### 5.3 Targets and objective functions
Define target sets `T_a` per stakeholder:
- Market: embeddings of job roles / skills
- Mission: mission statement(s)
- University: program outcomes / canonical curriculum texts
- Learner: learner profile vector

Define distances (MVP):
- cosine distance in lens space:
  - `d_a(option) = 1 - cos( L_a(v_option), centroid(L_a(T_a)) )`

Vector evaluation:
- `F(option) = { "market": d_market, "mission": d_mission, "university": d_university, "learner": d_learner, ... }`
Smaller is better (or invert for “fit”).

### 5.4 Pareto and planning
- Candidate set `C` for a given context (learner profile, constraints).
- Pareto set = nondominated options on selected objectives.

Short pathway planning (MVP):
- depth 2–3 with beam search:
  - state update: `v_{t+1} = (1-β)*v_t + β*v_course`
- objective aggregation: sum or discounted sum over steps.

### 5.5 Ethics / autonomy constraints
Replace “zombie risk” with neutral formal metric (MVP):
- `AutonomyDrift = ∑_t max(0, α_market(t) - α_self(t) - δ)`
or constraint on objective imbalance.

Feasibility:
- hard constraints: `AutonomyDrift ≤ τ`, fairness constraints, cost/time bounds
- or soft penalties (but keep Pareto as primary UI output).

### 5.6 Explanations & recourse
**Local explanation (cheap)**
- show per-objective contributions:
  - distance decomposition vs target centroid
  - nearest neighbors in each target set (retrieval-based “evidence”)

**Counterfactual / recourse**
- Goal: minimal change (option replacement or additional “bridge” module) that moves the solution to satisfy constraint or improve a chosen objective without breaking others too much.

Implementation tiers:
1) Tier‑0 (always): heuristic search over “bridge options” + report minimal changes.
2) Tier‑1 (optional): RobustX integration for robust counterfactuals.

---

## 6. API contract (FastAPI)

### Endpoints (minimal)
- `POST /ingest/run` — start ingestion job for a configured corpus
- `POST /embed/run` — embed new artifacts
- `POST /evaluate/options` — evaluate candidate options for a learner context
- `POST /pareto/options` — return Pareto set + tradeoff metadata
- `POST /plan/pathways` — generate short pathways (beam/greedy) + Pareto
- `POST /explain/option` — explanation payload for a selected option
- `POST /recourse/option` — counterfactual/recourse suggestions
- `POST /telemetry/event` — UI event logging for studies

### Response payload (Pareto)
- `pareto_options: list[{ option_id, objectives, feasible, violations, summary }]`
- `frontier_meta: { objectives_used, hypervolume(optional), counts }`

---

## 7. Engineering quality gates (for IJCAI reproducibility)
- **Pinned environments**: lockfile + docker image tag.
- **Determinism**: explicit seeds; store configs with each run.
- **Data provenance**: store `source_url`, `timestamp`, `raw_text_hash`.
- **Reports**: auto-generate tables/figures into `experiments/reports/`.
- **CI**: run unit tests + 1 smoke experiment on tiny dataset.

---

## 8. Implementation backlog (epics → tasks → acceptance)

### EPIC A — ingestion & artifact store
- A1: implement `ArtifactStore` (local FS) + hashing
- A2: MIT catalog/OCW ingestion (HTML → text normalization)
- A3: OpenAlex ingestion (title+abstract+topics)
- A4: O*NET/ESCO ingestion (roles, skills)

**Acceptance:** running `run_ingest.py --config mit_ocw_demo.yaml` creates a stable artifact dataset with provenance fields.

### EPIC B — embeddings
- B1: encoder interface + cache
- B2: embed artifacts + persist vectors
- B3: embedding versioning (model_id) and cache invalidation

**Acceptance:** `run_embed.py` produces embeddings and re-running doesn’t recompute unchanged items.

### EPIC C — lenses + objectives
- C1: diagonal lens + config for weights
- C2: target builders (centroid, set-of-points)
- C3: distance/objective functions

**Acceptance:** `run_eval.py` outputs `EvaluationVector` for a sample learner and candidate courses.

### EPIC D — Pareto/planning
- D1: Pareto nondominated sorting
- D2: pathway planning depth 2–3 (beam)
- D3: constraints integration

**Acceptance:** `run_ablation.py --case pareto_demo` generates a Pareto set with constraints toggled.

### EPIC E — explainability + recourse
- E1: nearest-neighbor evidence per objective
- E2: heuristic recourse (bridge modules)
- E3: RobustX adapter (optional)

**Acceptance:** UI can click an option and see “why” + “what if” suggestions.

### EPIC F — UI + telemetry
- F1: fork HypoCompass UI skeleton; replace pages with ADS dashboard
- F2: Pareto scatter + radar + 2D/3D projection views
- F3: contestability controls (constraints, objective selection)
- F4: telemetry logging

**Acceptance:** “single-session demo” works end-to-end and logs events.

### EPIC G — experiment harness
- G1: experiment runner (hydra/omegaconf)
- G2: baseline wrappers (pyKT/EduStudio, MONB, fair rank agg, FAiRDAS)
- G3: report builder (tables + figures)
- G4: ablation scripts

**Acceptance:** `make reproduce_small` regenerates all key figures on a toy dataset.

---

## 9. Notes on baselines integration

### KT baselines (EduStudio/pyKT)
- Keep them in `baselines/` with a small wrapper API:
  - `fit(dataset)`, `predict(state, action)`, `evaluate()`.
- ADS only consumes `predict` as an objective proxy.

### MONB baseline
- Map ADS “next step recommendation” to contextual bandit:
  - context = learner vector + constraints,
  - action = course,
  - reward vector = (market_fit, mission_fit, learning_gain, cost, fairness, …).

### FAiRDAS
- Reuse metric computation patterns and reporting (stability across perturbations).

### Fair rank aggregation
- Use as baseline “how to output a single list” from multi-objective scores.

---

## 10. Deliverables checklist (what colleague should produce)
- Working repo with `docker compose up` launching UI + API
- `experiments/` producing at least:
  - Pareto demo on open data
  - 1 outcome sanity-check
  - 1 fairness/stability report
- Reproducibility README: steps + expected outputs (hashes)
