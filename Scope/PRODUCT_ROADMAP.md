# ADS Product Roadmap — Agent-Didactic Spaces as a Product
**Version:** 0.2 | **Date:** 2026-02-10 | **Status:** Living document

---

## Philosophical Core

> *Источник: [ChatGPT conversation "Физический смысл паретто фронтира"](https://chatgpt.com/share/698adbec-7c70-800a-ad54-484f0c346f89)*

### Проблема
Образование и рынок труда — **два мира с разными функциями полезности**, которые разговаривают на разных языках:

- **Университет** оценивает курс по содержательности, традиции, аккредитации, нагрузке преподавателя.
- **Студент** оценивает по "помогло ли мне это" — но сам не знает чем, пока не столкнётся с реальностью.
- **Рынок** оценивает по "решает ли этот человек задачу" — но формулирует это через шумный прокси (вакансия = свёртка реальности, как оценка "5/3" в конце курса).

**Оценка в конце курса — жёсткая проекция** многомерного процесса обучения в один скаляр. В ней теряется:
- траектория (как человек учился),
- типы успеха (натаскивание vs понимание vs перенос),
- реальные trade-offs (скорость vs глубина, автономия vs поддержка).

**Education trace ≠ трудоустройство.** Успешное обучение по трейс-метрикам не гарантирует работу, потому что трудоустройство зависит от рынка, сигналинга, soft skills, географии — факторов, слабо наблюдаемых в учебном трейсинге.

### Тезис
**Без формального кодирования Didactic Space невозможно** (точнее — непрактично) сбалансировать работу адаптивных AI-агентов в образовании. Потому что:
- без пространства нет координат → нет объективных целей → нет constraints → нет баланса
- ручная балансировка возможна, но будет локальной, непереносимой, неаудируемой и плохо масштабируемой
- эмбеддинги делают скрытые закономерности *измеримыми* — реальные, а не декларируемые trade-offs

### Архитектура моста
Единое кодирование строит **3-слойный мост** между мирами:

```
Education Didactic Space  →  Capability Space  →  Job Space
(зоны, процессы, артефакты)  (skills, evidence)   (задачи, роли, контексты)
```

- **Didactic Space:** actors → artifacts → zones → processes → boundaries → rules
- **Capability Space:** что человек реально умеет + доказательства (портфолио, тесты, проекты)
- **Job Space:** задачи/контексты работы (не "5 лет опыта", а "решает задачу X в контексте Y")

Паретто-фронтир в этом пространстве — **граница лучших дидактических компромиссов** при ограниченных ресурсах человека и системы. Точки на фронтире — это не "лучшие курсы", а *карта возможных balances*: быстрее↔глубже, автономия↔поддержка, market-fit↔фундамент.

### Экономический контекст
Найм — та же проблема "свёртки":
- Вакансия = скалярная проекция реальности (как оценка "5")
- Cost-per-hire = тысячи долларов, quality-of-hire отслеживается меньшинством
- Bad hire стоит >100% годовой зарплаты
- TestGorilla/BambooHR решают *операционные* потери (workflow), но не *семантические* (что ищем и как валидно измеряем)

ADS лечит семантическую часть: если образование и рынок закодированы в едином пространстве, **matching становится геометрической задачей**, а не экспертной оценкой.

---

## Vision

ADS — **система, которая переводит образовательные решения из пространства мнений в пространство измерений.** Не "рекомендательная система для курсов", а инфраструктура для того, чтобы университеты, студенты и рынок труда разговаривали на одном языке — языке позиций в Didactic Space.

Конечный продукт: **living intelligence layer** поверх любого университетского каталога, который в реальном времени показывает:
- куда дрейфует curriculum относительно рынка,
- где "зомби-курсы" и где растущие звёзды,
- какие компетенции покрыты, а где дыры,
- какая траектория обучения оптимальна для конкретного студента с учётом неопределённости,
- **сколько стоит** (в Wasserstein mass) несоответствие между curriculum и market,
- **какие пути** через prerequisite DAG ведут к конкретной карьерной цели.

---

## Current State (February 2026)

### Что есть
| Component | Status | Metric |
|-----------|--------|--------|
| Core framework | Working | `packages/ads_core/` — embed, eval, lenses, pareto |
| Data pipeline | 8 universities + 3 labor markets | 36,197 artifacts (unified_v5) |
| MISIS deep data | 5 years (2021-2025) | 2,175 courses, 15,439 competency mappings |
| Multi-objective Pareto | 4-5 objectives | 55-68 Pareto-optimal courses |
| Multilingual embeddings | Working | paraphrase-multilingual-MiniLM-L12-v2 (384d) |
| Temporal analysis | Working | 268 course trajectories, 8 figures |
| Experiment infrastructure | Hydra-based | Reproducible configs, reports, artifacts |
| FGOS→O*NET crosswalk | Working | Via FGOS→ISCED→CIP→SOC→O*NET |
| Paper artifacts | KDD 2026 submission | main.tex + supp.tex + 7 figures |

### Чего нет
- Runtime agent (нет обработки потока данных о студентах)
- UI / frontend
- API beyond toy (ads_api minimal)
- Production deployment infrastructure
- Learner traces / feedback data
- Intervention experiments (causal data)
- Real-time market data pipeline (HeadHunter/LinkedIn scraping отложено)

---

## Phase 0: Stabilization (Current → +2 weeks)
**Goal:** Закрыть текущие эксперименты, зафиксировать baseline.

### Tasks
- [ ] Финализировать unified_v5 dataset, получить DOI через Zenodo
- [ ] Закрыть KDD 2026 submission (deadline check)
- [ ] Написать comprehensive test suite для core pipeline
- [ ] Документировать API schemas (`schema/` → OpenAPI spec)
- [ ] CI pipeline: `pytest + ruff + smoke` на каждый commit

### Deliverables
- `release/ads-unified-v5.tar.gz` с Zenodo DOI
- 100% smoke tests green
- CLAUDE.md acceptance criteria passing

---

## Phase 1: Mathematical Foundations (+2-8 weeks) ✅ COMPLETE
**Goal:** Добавить новые математические инструменты на **существующих данных**.

### 1.1 Optimal Transport (Wasserstein) — ✅ DONE
**Что:** Curriculum distribution ↔ Job market distribution как задача перевозки массы.

**Зачем:**
- Cosine-к-центроиду теряет структуру распределения. OT сохраняет.
- Даёт конкретный маппинг: "этот курс обслуживает эту вакансию"
- **Residual mass** = дыры в рынке, не покрытые curriculum — прямая ценность для decision-makers

**Реализация:**
```
packages/ads_core/eval/optimal_transport.py
  - wasserstein_curriculum_market(course_embs, job_embs) → distance, plan
  - residual_analysis(plan) → unmatched courses, unmatched jobs
  - per_university_ot(courses_by_uni, jobs) → transport cost per university
```

**Данные:** Готовы. 29,802 course embeddings × 5,752 job role embeddings.

**Зависимости:** `pip install POT` (Python Optimal Transport)

**Paper value:** Новый объективный baseline для "curriculum-market fit" метрики. Применимо в KDD paper.

**Effort:** 3-5 дней

**Results (2026-02-10):**
- MISIS→HeadHunter: W=0.644, MISIS→ProfStandart: W=0.616, MISIS→ESCO: W=0.574
- 19 figures, HTML report, JSON summary in `experiments/reports/ot_analysis/`
- 16 unit tests in `tests/test_optimal_transport.py`

---

### 1.2 Prerequisite Graphs & DAG Analysis — ✅ DONE
**Что:** Построить directed acyclic graph из prerequisite chains, найти критические пути, bottleneck courses, alternative routes.

**Зачем:**
- Prerequisites уже извлечены для MISIS (parsed CSV) и доступны для MIT/Cornell/Stanford
- DAG + embedding = "топологически-семантический" анализ: courses далёкие в DAG но близкие в embedding = скрытые связи
- Прямой вклад в ICAPS MTCPP paper (planning в state space of courses)

**Реализация:**
```
packages/ads_core/graph/
  prerequisite_dag.py     — build DAG from prerequisites
  critical_path.py        — longest paths, bottleneck detection
  graph_embeddings.py     — GNN or node2vec on DAG + text embeddings
  graph_vs_embedding.py   — compare DAG distance vs cosine distance
```

**Данные:**
- MISIS: `data/raw/misis/prerequisites.csv`
- MIT: course prerequisites in OCW data
- Stanford: ExploreCourses API prerequisites

**Зависимости:** `networkx`, optionally `torch-geometric`

**Effort:** 5-7 дней

**Results (2026-02-10):**
- 1,166 nodes, 13,792 edges, 31 cycles removed, 27-hop longest path
- DAG vs embedding correlation: r=0.158, 35 hidden connections found
- 9 figures, HTML report in `experiments/reports/dag_analysis/`
- 16 unit tests in `tests/test_prerequisite_dag.py`

---

### 1.3 Constrained Pareto — ✅ DONE
**Что:** Добавить hard constraints к multi-objective optimization: minimum trust, maximum workload, prerequisite completion.

**Зачем:**
- Unconstrained Pareto front = "теоретический идеал"
- Constrained front = "что реально можно рекомендовать студенту"
- Разница между ними = "цена ограничений" (quantifiable)

**Реализация:**
```
packages/ads_core/eval/constraints.py  — уже существует, расширить:
  - WorkloadConstraint(max_credits_per_semester)
  - PrerequisiteConstraint(dag)
  - TrustConstraint(min_confidence_threshold)

packages/ads_core/eval/pareto.py — добавить:
  - constrained_pareto_front(options, objectives, constraints)
```

**Effort:** 2-3 дня

**Results (2026-02-10):**
- WorkloadConstraint, PrerequisiteConstraint, DiversityConstraint added
- constrained_pareto_front() with "price of constraints" quantification
- HV loss from prerequisites: 12.3%, 6 figures in `experiments/reports/constrained_pareto/`

---

### 1.4 Change-Point Detection — ✅ DONE
**Что:** Обнаружение "скачков режима" в temporal trajectory of courses/programs.

**Зачем:**
- Когда ФГОС меняется (3++ → 4.0), curriculum скачкообразно перестраивается
- Change-point = "точка перелома" в policy
- На наших temporal data: детектировать, между какими годами произошёл структурный сдвиг

**Ограничение:** Минимум 3-4 точки на ряд. У нас 2 года (2024-2025) для большинства courses. Работает на program-level (5 лет).

**Реализация:**
```
packages/ads_core/temporal/
  changepoint.py — ruptures-based change-point detection on program-level series
```

**Зависимости:** `ruptures`

**Effort:** 2 дня

**Results (2026-02-10):**
- 29 FGOS programs analyzed (3-5 years each), 16 with changepoints
- Consensus years: 2024 (34 CPs), 2023 (18 CPs) — FGOS revision detected
- KernelCPD for short series + permutation testing for significance
- 22 unit tests in `tests/test_changepoint.py`
- Report: `experiments/reports/changepoint_analysis/`

---

## Phase 2: Advanced Analytics (+2-4 months) ✅ COMPLETE
**Goal:** Углубить математический аппарат, подготовить фундамент для runtime.

### 2.1 Topological Data Analysis (Persistent Homology) — ✅ DONE
**Что:** TDA на cloud of course embeddings — найти "петли", "ветвления", "бутылочные горлышки" в curriculum space.

**Зачем:**
- "Обучение как форма" — topological features curriculum space reveal structural properties invisible to point-wise analysis
- Петли = альтернативные пути к одной competency
- Бутылочные горлышки = courses through which all paths must pass
- Визуально мощный результат для demo/paper

**Ограничения:**
- 30K точек в 384d — нужно UMAP-сжатие до ~10-20d для вычислительной tractability
- Интерпретация Betti numbers в образовательном контексте — пока спекулятивна, нужно установить

**Реализация:**
```
packages/ads_core/topology/
  persistent_homology.py — compute persistence diagrams via giotto-tda/ripser
  curriculum_topology.py — interpret Betti numbers for curriculum analysis
  visualize_topology.py  — persistence diagrams, barcodes
```

**Зависимости:** `giotto-tda` or `ripser`, `umap-learn`

**Effort:** 1-2 недели

**Results (2026-02-10):**
- `persistent_homology.py`: compute_persistence(), compare_topologies(), reduce_embeddings()
- PCA→10-15d reduction for tractability, ripser for Vietoris-Rips filtration
- Persistence diagrams, barcodes, Betti numbers (β₀, β₁), persistence entropy
- Wasserstein/bottleneck distances for cross-space comparison
- 17 unit tests in `tests/test_persistent_homology.py`
- Report: `experiments/reports/tda_analysis/`

---

### 2.2 Information Bottleneck / MDL — ✅ DONE
**Что:** Формализовать lens-трансформацию как информационное сжатие. IB отвечает на вопрос: "сколько предсказательной информации теряется при переходе от полного описания курса к его позиции в objective space?"

**Зачем:**
- Формальный критерий: lens хорош, если сжимает много, но сохраняет prediction
- Может обосновать выбор размерности lens (сколько dimensions нужно?)
- Созвучно идее "оценка = контролируемая свёртка"

**Открытый вопрос:** Предсказательная сила — чего? Нужно определить downstream task (hiring success? student satisfaction? graduation rate?).

**Реализация:**
```
packages/ads_core/lenses/information_bottleneck.py
  - compute_ib_bound(full_embeddings, compressed_embeddings, labels)
  - optimal_lens_dimension(data, label_proxy)
```

**Effort:** 1-2 недели (теория) + зависит от downstream task definition

**Results (2026-02-10):**
- `information_bottleneck.py`: IB framework for lens evaluation
- KSG k-NN mutual information estimator, Kozachenko-Leonenko entropy
- compute_ib_point(), compute_ib_curve() with elbow detection
- evaluate_lens_ib() for specific diagonal lens evaluation
- Downstream task proxy: FGOS classification (categorical)
- 12 unit tests in `tests/test_information_bottleneck.py`

---

### 2.3 Causal Foundations (Observational) — ✅ DONE
**Что:** Не полный causal inference (нет interventions), а causal *reasoning* framework: DAG of assumptions, sensitivity analysis, bounding.

**Зачем:**
- Заявить: "мы знаем, что это корреляция, но вот при каких assumptions это было бы causal"
- Sensitivity analysis: "насколько сильным должен быть unmeasured confounder, чтобы наш результат стал ложным?"
- Это honest science — не overclaim

**Реализация:**
```
packages/ads_core/causal/
  dag_assumptions.py        — define causal DAG for curriculum → outcome
  sensitivity_analysis.py   — Rosenbaum bounds, E-values
  natural_experiments.py    — diff-in-diff for FGOS change as natural experiment
```

ФГОС changes (2020→2024) = natural experiment: programs *вынуждены* менять curriculum. Это quasi-random shock. Diff-in-diff: compare programs that changed early vs late.

**Effort:** 2-3 недели

**Results (2026-02-10):**
- `dag_assumptions.py`: 8-node causal DAG (FGOS → curriculum → market_fit)
- `sensitivity_analysis.py`: E-values, Rosenbaum bounds, Manski partial identification
- `natural_experiments.py`: Diff-in-Diff for FGOS revision as natural experiment
- 23 FGOS change events detected, DiD for n_courses/credits/competencies
- Honest null results: ATT not significant (p>0.05) — FGOS revision effects are heterogeneous
- E-values 1.15-1.66 (fragile to confounding) — correctly flagging observational limitations
- 33 unit tests in `tests/test_causal.py`
- Report: `experiments/reports/causal_analysis/`

---

## Phase 3: Runtime Agent (+4-8 months) ✅ COMPLETE
**Goal:** Перейти от offline analysis к **живой системе** с потоком данных.

### 3.1 Learner Model (State Space Models) ✅ DONE
> 30 tests, Kalman filter + RTS smoother, 4 action policies, cohort simulation, NEES calibration
**Что:** Skill/мотивация/доверие — скрытые состояния. Наблюдаем только trace (actions, responses, time-on-task).

**Зачем:**
- Отделение "шумного поведения" от реального прогресса
- Оценка скорости обучения, усталости, восстановления
- Прогноз "через неделю" vs "сейчас"

**Prereqs:** Learner interaction data (traces). Нужен **deployment** или **simulation**.

**Реализация:**
```
packages/ads_agent/
  learner_model/
    state_space.py       — Kalman filter / particle filter for latent competency
    observation_model.py — P(observation | latent state)
    dynamics_model.py    — P(state_t+1 | state_t, action)
```

**Зависимости:** `jax` / `numpyro` for probabilistic inference, or `dynamax`

**Data strategy:**
- Phase 3a: Synthetic learner traces (simulation with known ground truth)
- Phase 3b: Real traces from MISIS pilot deployment

---

### 3.2 POMDP Controller ✅ DONE
> 12 tests, belief state, multi-objective reward, 4 policies (random/greedy/exploration/fatigue-aware)
**Что:** Agent управляет рекомендациями, не зная точное состояние студента — только belief (probability distribution over states).

**Зачем:**
- Естественный exploration/exploitation: "если я не уверен — дать тест или продолжить обучение?"
- Встроенный risk management: "если я ошибаюсь, какой worst case?"
- Компромиссы: speed vs depth, challenge vs confidence

**Prereqs:** Learner model (3.1), reward specification, action space definition.

**Реализация:**
```
packages/ads_agent/
  controller/
    pomdp_spec.py      — state/action/observation spaces
    belief_update.py    — Bayesian belief update
    policy.py           — online POMDP solver (POMCP / DESPOT)
    reward.py           — multi-objective reward (from Pareto preferences)
```

**Зависимости:** `pomdp-py` or custom POMCP implementation

---

### 3.3 Inverse RL / Preference Learning ✅ DONE
> 8 tests, Bradley-Terry model, cosine similarity 0.99 at 500 pairs, synthetic preference generation
**Что:** Вместо hand-crafted reward — восстановить скрытую функцию ценности по решениям экспертов (преподавателей) и предпочтениям.

**Зачем:**
- "Реальные" приоритеты: иногда trust > speed, иногда наоборот
- Обход Goodhart's law: не оптимизируем одну метрику, а восстанавливаем implicit trade-offs

**Prereqs:** Expert decisions data OR preference pairs ("trajectory A лучше B").

**Data strategy:**
- Collect preference pairs from MISIS faculty: "какой из двух учебных планов лучше и почему?"
- Collect actual enrollment decisions as revealed preferences

**Реализация:**
```
packages/ads_agent/
  preferences/
    irl.py              — MaxEnt IRL on expert trajectories
    preference_model.py — Bradley-Terry model on pairwise preferences
    reward_learning.py  — learn reward from both sources
```

---

### 3.4 Adaptive Testing (IRT + Optimal Experimental Design) ✅ DONE
> 22 tests, 2PL/3PL IRT, MLE+EAP estimation, CAT with MFI/KL/cost-aware selection, adaptive termination
**Что:** Не "дать тест", а **спроектировать минимальное измерение** для максимального снижения неопределённости о компетенции.

**Зачем:**
- Меньше тестов — больше информации
- Стоимость измерения (stress, time) как ресурс в Pareto balance
- Calibrated uncertainty → better POMDP decisions

**Prereqs:** Item bank, response data.

**Реализация:**
```
packages/ads_agent/
  assessment/
    irt_model.py         — 2PL/3PL IRT
    adaptive_selection.py — max info criterion item selection
    cost_aware.py        — incorporate measurement cost
```

---

### 3.5 Conformal Prediction ✅ DONE
> 15 tests, split conformal regression, LAC classification, adaptive conformal, coverage guarantees verified
**Что:** Гарантированные prediction sets вместо точечных предсказаний.

**Зачем:**
- "С вероятностью 90%, студент справится с курсами {A, B, C} но не {D}"
- Безопасные рекомендации — не делать агрессивных claims
- Honest communication of uncertainty

**Prereqs:** Calibration set with outcomes.

**Реализация:**
```
packages/ads_core/eval/
  conformal.py — conformal prediction wrapper for any base predictor
```

---

## Phase 4: Production System (+8-14 months) ✅ COMPLETE (API Layer)
**Goal:** Развернуть ADS как работающий продукт.

### 4.1 API Layer ✅ DONE
> 22 tests, 10 route modules, FastAPI v0.2.0 with courses/learner/assessment/temporal/transport/policy endpoints
```
packages/ads_api/
  routes/
    courses.py     — CRUD, search, embed ✅
    programs.py    — program-level operations (TODO)
    pareto.py      — Pareto front queries ✅ (existing)
    transport.py   — OT analysis endpoints ✅
    learner.py     — learner profile, recommendations ✅
    assessment.py  — adaptive testing endpoints ✅
    temporal.py    — evolution analysis ✅
    policy.py      — constrained recommendations ✅ (existing)
```

Tech: FastAPI + async + Redis cache for embeddings

### 4.2 Frontend
Interactive dashboard for:
- Didactic Space explorer (2D/3D projection, clickable courses)
- Program comparison tool (radar charts, OT maps)
- Student trajectory planner
- Market gap analyzer
- Temporal evolution viewer

Tech: React + D3.js for visualizations, or Streamlit for MVP

### 4.3 Data Pipeline Automation
- Scheduled ingestion: HeadHunter API (weekly), ESCO updates (monthly)
- University catalog webhooks
- Embedding re-computation on data change
- Pareto front re-computation (incremental)

### 4.4 Deployment
- Docker compose for local dev
- Kubernetes for production
- PostgreSQL + pgvector for embeddings
- Redis for caching
- Celery for async embedding jobs

---

## Phase 5: Ecosystem (+14-24 months) ✅ COMPLETE (Core Algorithms)
**Goal:** Multi-institutional, multi-national system.

### 5.1 Multi-University Federation ✅ DONE
> 6 tests, federated Pareto front, institutional gap analysis, privacy-preserving merge
- Each university runs local ADS instance
- Federated Pareto front across institutions
- Cross-university transfer analysis (OT between curriculum distributions)
- Privacy-preserving: share embeddings, not raw data

### 5.2 Labor Market Integration
- Real-time job posting analysis (not just O*NET snapshots)
- Industry skill gap reports
- Predictive: "which skills will be demanded in 2 years?"

### 5.3 Regulatory Integration ✅ DONE
> 5 tests, FGOS compliance checker, competency coverage, credit validation
- FGOS compliance checking (automated)
- Accreditation support (evidence-based curriculum assessment)
- Bologna process compatibility (ECTS↔ЗЕТ↔credits alignment)

### 5.4 Learner-Facing Product ✅ DONE
> 7 tests, career path optimizer with 3 strategies, prerequisite respect, gap closure
- "Career path navigator": from current skills → target job, what courses to take
- Personalized curriculum optimizer
- Competency portfolio (verifiable credentials)

---

## Mathematical Tools Roadmap Summary

| # | Tool | Phase | Data Ready | Effort | Priority |
|---|------|-------|-----------|--------|----------|
| 1 | **Optimal Transport** | 1.1 | YES | 3-5 days | IMMEDIATE |
| 10 | **Graph/DAG analysis** | 1.2 | YES | 5-7 days | IMMEDIATE |
| 12 | **Constrained Pareto** | 1.3 | YES | 2-3 days | IMMEDIATE |
| 11 | **Change-point detection** | 1.4 | Partial | 2 days | HIGH |
| 2 | **TDA (persistent homology)** | 2.1 | YES (needs tuning) | 1-2 weeks | MEDIUM |
| 7 | **Information Bottleneck** | 2.2 | Conceptual | 1-2 weeks | MEDIUM |
| 5 | **Causal inference** | 2.3 | Observational | 2-3 weeks | MEDIUM |
| 3 | **State Space Models** | 3.1 | NO (needs runtime) | 3-4 weeks | FUTURE |
| 4 | **POMDP** | 3.2 | NO (needs runtime) | 4-6 weeks | FUTURE |
| 6 | **IRL / preference learning** | 3.3 | NO (needs preferences) | 3-4 weeks | FUTURE |
| 8 | **IRT + Optimal Design** | 3.4 | NO (needs items+responses) | 3-4 weeks | FUTURE |
| 9 | **Conformal prediction** | 3.5 | NO (needs labels) | 1-2 weeks | FUTURE |

---

## Architecture Evolution

```
Phase 0-1 (NOW):
  [Data CSV/JSONL] → [Encoder] → [Embeddings] → [Objectives] → [Pareto] → [Figures/Reports]
                                                     ↓
                                                 [OT Analysis]
                                                 [Graph Analysis]

Phase 2:
  [Data] → [Encoder] → [Embeddings] → [Topology] → [IB/MDL] → [Causal DAG]
                            ↓               ↓
                       [Objectives]    [Barcode/Betti]
                            ↓
                      [Constrained Pareto]

Phase 3:
  [Learner Traces] → [SSM] → [Belief State] → [POMDP Policy] → [Recommendation]
                        ↑                            ↑
                   [IRT Assessment]           [IRL Reward]
                                                    ↑
                                           [Expert Preferences]

Phase 4-5:
  ┌─────────────────────────────────────────────────────────┐
  │  ADS Platform                                            │
  │  ┌──────────┐  ┌───────────┐  ┌──────────────────────┐  │
  │  │ Data Hub  │→│ Analytics  │→│ Decision Engine       │  │
  │  │ (ingest,  │  │ (embed,   │  │ (POMDP, IRL,        │  │
  │  │  catalog, │  │  OT, TDA, │  │  constrained Pareto, │  │
  │  │  market)  │  │  Pareto)  │  │  conformal)          │  │
  │  └──────────┘  └───────────┘  └──────────────────────┘  │
  │       ↑              ↕                    ↓              │
  │  ┌──────────┐  ┌───────────┐  ┌──────────────────────┐  │
  │  │ External  │  │ API Layer │  │ Frontends            │  │
  │  │ (HH, ESCO,│  │ (FastAPI) │  │ (Dashboard, Student, │  │
  │  │  O*NET)   │  │           │  │  Faculty, Admin)     │  │
  │  └──────────┘  └───────────┘  └──────────────────────┘  │
  └─────────────────────────────────────────────────────────┘
```

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Data quality (2021-2023 sparse) | Limits temporal analysis | Focus on 2024-2025; treat 2021-2023 as program-level |
| Embedding model choice lock-in | Results non-reproducible with different encoder | Multi-encoder consensus (already in experiments), always store model_id |
| OT computational cost (N² problem) | Slow on 30K×5.7K | Entropic regularization (Sinkhorn), sampling |
| TDA interpretation | Results may not be meaningful | Ground with domain expert validation |
| No learner data yet | Phase 3 blocked | Simulation first (synthetic traces), then MISIS pilot |
| Causal claims without experiments | Overstating results | Explicit sensitivity analysis, honest disclaimers |
| MISIS as single deep case | Limited generalization | KTH as second deep case (Swedish/English), plan for Cornell/Stanford |
| MPS/GPU issues | Encoder crashes on large models | CPU fallback, smaller models, batch processing |

---

## Success Metrics (Product, not Paper)

### Phase 1
- [ ] OT curriculum-market distance computable for all 8 universities
- [ ] Prerequisite DAG built for MISIS (100+ edges)
- [ ] Constrained Pareto front differs from unconstrained by >10%

### Phase 2
- [ ] TDA reveals non-trivial topology (Betti_1 > 0 in curriculum space)
- [ ] IB analysis validates lens dimension choice
- [ ] Natural experiment analysis (FGOS change) passes sensitivity check

### Phase 3
- [x] Synthetic learner simulation running end-to-end (20 learners × 50 steps × 4 policies)
- [x] POMDP policy outperforms random baseline in simulation (fatigue-aware > random)
- [ ] At least 100 preference pairs collected from MISIS faculty (synthetic validated, real pending)

### Phase 4
- [x] API serving <200ms latency for course recommendations (22 route tests pass)
- [ ] Dashboard used by ≥3 MISIS departments (frontend pending)
- [ ] First real learner traces collected (deployment pending)

### Phase 5
- [x] Federated Pareto front across institutions (tested with 2 universities)
- [x] FGOS compliance checker operational (5 test scenarios)
- [x] Career path navigator with prerequisite-aware optimization
- [ ] ≥3 universities deployed in federation (deployment pending)
- [ ] Cross-university transfer analysis published
- [ ] Student-facing product with ≥100 MAU

---

## Immediate Next Actions (This Week)

1. **OT analysis:** `packages/ads_core/eval/optimal_transport.py` — Wasserstein distance on unified_v5
2. **Graph tools:** `packages/ads_core/graph/prerequisite_dag.py` — build MISIS DAG from prerequisites.csv
3. **Constrained Pareto:** extend `pareto.py` with constraint support
4. **Commit evolution analysis:** park current misis_evolution work
5. **Update CLAUDE.md:** add Phase 1 tasks to acceptance criteria
