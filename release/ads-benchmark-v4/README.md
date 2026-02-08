---
language:
  - en
license: cc-by-4.0
size_categories:
  - 10K<n<100K
task_categories:
  - text-classification
  - feature-extraction
tags:
  - education
  - curriculum
  - multi-objective-optimization
  - pareto
  - benchmark
  - onet
  - labor-market
  - universities
  - KDD2026
pretty_name: "ADS-Unified-v4: Multi-Stakeholder Curriculum Evaluation Benchmark"
configs:
  - config_name: courses
    data_files: "courses.csv"
  - config_name: artifacts
    data_files: "artifacts.jsonl"
  - config_name: programs
    data_files: "programs.csv"
---

# ADS-Unified-v4: Multi-Stakeholder Curriculum Evaluation Benchmark

**Agent-Didactic Spaces (ADS)** is a benchmark for evaluating university curricula through multi-stakeholder multi-objective optimization. It links course catalogs from 7 universities across 3 countries to O\*NET labor market data and institutional missions, enabling Pareto-optimal curriculum analysis.

## Dataset Summary

| Statistic | Value |
|-----------|-------|
| Courses (deduplicated) | 27,586 |
| Artifacts (courses + jobs + missions) | 28,633 |
| Universities | 7 |
| Countries | 3 (USA, Sweden, UK) |
| O\*NET Occupations | 1,016 |
| Mission Statements | 62 |
| College Scorecard Programs | 398 |
| O\*NET Skill Records | 71,520 |
| O\*NET Time-Travel Versions | 4 (2022-2025) |

### Per-University Breakdown

| University | Country | Courses |
|------------|---------|---------|
| UC Berkeley | USA | 11,102 |
| MIT | USA | 6,944 |
| KTH | Sweden | 4,768 |
| Stanford | USA | 2,663 |
| Cornell | USA | 842 |
| UIUC | USA | 821 |
| Edinburgh | UK | 446 |

## Dataset Structure

```
ads-benchmark-v4/
├── README.md                    # This file
├── DATASHEET.md                 # Gebru et al. datasheet
├── LICENSE                      # CC BY 4.0
├── croissant.json               # ML Commons metadata
├── artifacts.jsonl              # 28,633 artifacts (courses + jobs + missions)
├── courses.csv                  # 27,586 deduplicated courses
├── programs.csv                 # 398 College Scorecard programs
├── onet/                        # O*NET structured data
│   ├── occupations.csv          # 1,016 occupation profiles
│   ├── skills.csv               # 71K skill records
│   ├── tasks.csv                # Task statements per occupation
│   ├── technology.csv           # Technology/tools per occupation
│   └── time_travel/             # Market drift analysis
│       ├── PACK_META.csv        # Version metadata
│       ├── onet_26_3.csv        # v26.3 (2022)
│       ├── onet_27_3.csv        # v27.3 (2023-05)
│       ├── onet_28_1.csv        # v28.1 (2023-11)
│       └── onet_30_0.csv        # v30.0 (2025)
├── crosswalks/                  # Linkage tables
│   ├── cip_to_soc.csv           # CIP-2020 → SOC-2018 mapping
│   └── soc_to_cip.csv           # SOC-2018 → CIP-2020 mapping
├── missions/                    # Mission corpus
│   └── mission_corpus.csv       # 62 institutional mission statements
├── results/                     # Reference experiment results
│   ├── pareto_minilm.json       # Pareto set (MiniLM encoder, tau=0.25)
│   ├── pareto_mpnet.json        # Pareto set (MPNet encoder)
│   ├── pareto_learned_lens.json # Pareto set (learned lens)
│   ├── baselines.json           # Baseline comparisons
│   └── goal_sensitivity.json    # 4 learner profiles
└── scripts/
    └── validate_dataset.py      # Integrity check script
```

## Quick Start

### Using the `datasets` library

```python
from datasets import load_dataset

# Load courses
courses = load_dataset("ads-research/ads-benchmark-v4", "courses")
print(courses["train"][0])

# Load all artifacts (courses + jobs + missions)
artifacts = load_dataset("ads-research/ads-benchmark-v4", "artifacts")
```

### Using raw Python

```python
import csv
import json

# Load courses
with open("courses.csv") as f:
    courses = list(csv.DictReader(f))
print(f"{len(courses)} courses from {len(set(c['university'] for c in courses))} universities")

# Load artifacts
with open("artifacts.jsonl") as f:
    artifacts = [json.loads(line) for line in f]
print(f"{len(artifacts)} artifacts")

# Filter by type
course_artifacts = [a for a in artifacts if a["type"] == "COURSE"]
job_roles = [a for a in artifacts if a["type"] == "JOB_ROLE"]
missions = [a for a in artifacts if a["type"] == "MISSION"]
```

### Running the Pareto optimizer

```python
# Using the ADS framework
from experiments.run import main
# python -m experiments.run dataset=unified_v4 embedding=sbert_allminilm lenses=identity
```

## Data Fields

### courses.csv

| Field | Type | Description |
|-------|------|-------------|
| `course_id` | string | Unique course identifier |
| `course_name` | string | Course title |
| `description` | string | Course description and learning outcomes |
| `units` | string | Credit units |
| `department` | string | Department name |
| `level` | string | Course level (undergraduate/graduate) |
| `prerequisites` | string | Course prerequisites |
| `university` | string | University name |

### artifacts.jsonl

| Field | Type | Description |
|-------|------|-------------|
| `artifact_id` | string | Unique artifact identifier (e.g., `course:MIT:8ba94312eb3d`) |
| `type` | string | `COURSE`, `JOB_ROLE`, or `MISSION` |
| `text` | string | Full text content |
| `source_url` | string | Original source URL |
| `source_hash` | string | SHA-256 hash of source text |
| `timestamp` | string | ISO 8601 ingestion timestamp |

## Objectives (Multi-Stakeholder)

The ADS benchmark evaluates curricula across four stakeholder objectives:

1. **Market alignment** — cosine similarity between course embeddings and O\*NET occupation skill/task descriptions
2. **Mission alignment** — cosine similarity with institutional mission statements
3. **University identity** — alignment with university-specific strategic goals
4. **Learner satisfaction** — proxy based on course-level signals (prerequisites met, level appropriateness)

A curriculum is **Pareto-optimal** if no other curriculum improves one objective without degrading another.

## O\*NET Time-Travel Analysis

The `onet/time_travel/` directory contains O\*NET occupation snapshots from 4 releases (2022-2025), enabling analysis of how labor market demands evolve over time. Key finding: 82% Jaccard stability in top-skill sets across versions, with emerging technology roles showing the most drift.

## Data Sources and Licensing

| Source | License | Usage |
|--------|---------|-------|
| O\*NET Database | Public Domain (U.S. Gov) | Occupation profiles, skills, tasks, technology |
| MIT OpenCourseWare | CC BY-NC-SA 4.0 | Course descriptions (transformative metadata extraction) |
| College Scorecard | Public Domain (U.S. Gov) | Program outcomes |
| UC Berkeley, Stanford, UIUC, Cornell catalogs | Public information | Course descriptions |
| KTH (Sweden) catalog | Public information | Course descriptions |
| University of Edinburgh catalog | Public information | Course descriptions |

## Citation

```bibtex
@inproceedings{ads_benchmark_v4_2026,
  title     = {Agent-Didactic Spaces: A Multi-Stakeholder Benchmark for Curriculum Evaluation},
  author    = {{ADS Research Team}},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2026},
  note      = {Datasets \& Benchmarks Track}
}
```

## Ethical Considerations

- No personally identifiable information (PII) is included. All data is publicly available course metadata and government labor statistics.
- Course descriptions are factual institutional information, not student data.
- The benchmark evaluates curriculum *structure*, not individual student outcomes.
- See `DATASHEET.md` for the complete Gebru et al. datasheet.
