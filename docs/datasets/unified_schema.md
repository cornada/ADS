# Unified Dataset Schema for ADS

## Overview

This document defines the canonical schema for all datasets in the Agent-Didactic Spaces (ADS) system. The unified schema ensures consistency across institutions (MIT, UC Berkeley, ASU) and enables reproducible cross-dataset comparisons.

## Entity Types

### ArtifactType Enumeration

| Type | Description | Example |
|------|-------------|---------|
| `MISSION` | Institutional/departmental mission statement | MIT Mission, EECS Department Mission |
| `COURSE` | Course catalog entry or OCW course | 6.006 Introduction to Algorithms |
| `SYLLABUS` | Detailed course syllabus | Course outline with topics/schedule |
| `LAB` | Laboratory or hands-on component | 6.004 Lab assignments |
| `PAPER_ABSTRACT` | Research paper abstract | OpenAlex paper abstract |
| `JOB_ROLE` | Occupation/job role description | O*NET Software Developer |
| `SKILL` | Skill or competency description | O*NET Python Programming |
| `LEARNER_PROFILE` | Student profile/preferences | Target learner specification |
| `OUTCOME_MAJOR` | Career outcomes by major | UCB CS employment rate |
| `OUTCOME_SUMMARY` | Aggregated outcomes report | ASU college-level outcomes |
| `SURVEY_SCHEMA` | Survey instrument structure | ASU FDS survey questions |

## Canonical Artifact Schema

### Required Fields

```python
class Artifact:
    artifact_id: str       # Stable ID with institution prefix
    type: ArtifactType     # One of the types above
    text: str              # Normalized plain text for embeddings
    metadata: Dict         # Additional structured data
```

### Artifact ID Format

```
<source>:<identifier>

Examples:
- mit_catalog:6.006
- ocw:6.0001
- ucb_outcome:2024:COE:Computer_Science
- asu_fds:schema
- onet:15-1252.00
```

### Metadata Fields

| Field | Required | Description |
|-------|----------|-------------|
| `institution` | Yes | Institution code: `MIT`, `UCB`, `ASU`, `ONET` |
| `source_url` | Yes | Original source URL |
| `license` | Yes | License/terms (e.g., "CC BY-NC-SA 4.0") |
| `provenance` | Yes | Provenance tracking dict |

#### Provenance Sub-fields

```python
provenance = {
    "source_url": str,        # Original URL
    "fetched_at": datetime,   # ISO timestamp
    "raw_text_hash": str,     # SHA-256 of raw text
    "snapshot_path": str,     # Optional local snapshot
}
```

### Type-Specific Metadata

#### COURSE Artifacts

```python
metadata = {
    "institution": "MIT",
    "course_number": "6.006",
    "title": "Introduction to Algorithms",
    "department": "6",
    "source_url": "https://...",
    "license": "CC BY-NC-SA 4.0",
    "provenance": {...}
}
```

#### OUTCOME_MAJOR Artifacts

```python
metadata = {
    "institution": "UCB",
    "year": 2024,
    "college": "COE",
    "major": "Computer Science",
    "employment_rate": 0.85,
    "grad_school_rate": 0.12,
    "median_salary": 130000,
    "n_respondents": 450,
    "source_url": "https://...",
    "license": "Public aggregate data",
    "provenance": {...}
}
```

#### OUTCOME_SUMMARY Artifacts

```python
metadata = {
    "institution": "ASU",
    "college": "FSE",
    "college_name": "Fulton Schools of Engineering",
    "academic_year": "2015-2016",
    "employment_rate": 0.72,
    "grad_school_rate": 0.18,
    "career_outcomes_rate": 0.90,
    "median_salary": 62000,
    "source_url": "https://...",
    "license": "Public report",
    "provenance": {...}
}
```

#### SURVEY_SCHEMA Artifacts

```python
metadata = {
    "institution": "ASU",
    "version": "1.0",
    "section_count": 6,
    "question_count": 16,
    "source_url": "https://...",
    "license": "Public document",
    "provenance": {...}
}
```

## Outcomes Normalization

### Canonical Outcome Categories

| Category | Description | UCB Mapping | ASU Mapping |
|----------|-------------|-------------|-------------|
| `employed` | Full-time employment | `employment_rate` | `employment_rate` |
| `grad_school` | Graduate/professional school | `grad_school_rate` | `grad_school_rate` |
| `career_outcomes` | Employed + grad school | Computed | `career_outcomes_rate` |
| `seeking` | Still seeking employment | Computed | `seeking_rate` |

### Outcome Record Schema

```python
OutcomeRecord = {
    "institution": str,      # MIT, UCB, ASU
    "unit": str,             # College/department code
    "unit_name": str,        # Full name
    "cohort_year": str,      # Academic year
    "category": str,         # Canonical category
    "value": float,          # Rate (0-1) or amount
    "n": int,                # Sample size
    "source_url": str,
    "raw_hash": str,
}
```

## Dataset Bundle Structure

Each institution's data is packaged as a `DatasetBundle`:

```python
@dataclass
class DatasetBundle:
    dataset_id: str                    # "mit", "ucb", "asu", "toy"
    artifacts: List[Artifact]          # All artifacts
    outcomes: Optional[pd.DataFrame]   # Normalized outcomes table
    metadata: Dict[str, Any]           # Dataset-level metadata
```

### Dataset Metadata

```python
metadata = {
    "institution": str,
    "version": str,
    "created_at": datetime,
    "artifact_count": int,
    "artifact_types": List[str],
    "source_manifest": str,
}
```

## File Formats

### artifacts.jsonl

One JSON object per line:

```json
{"artifact_id": "mit_catalog:6.006", "type": "COURSE", "text": "...", "metadata": {...}}
{"artifact_id": "ocw:6.0001", "type": "COURSE", "text": "...", "metadata": {...}}
```

### outcomes_*.csv

Normalized CSV with standard columns:

```csv
institution,unit,unit_name,cohort_year,employment_rate,grad_school_rate,median_salary,n_respondents
UCB,COE,College of Engineering,2024,0.85,0.12,130000,450
ASU,FSE,Fulton Schools of Engineering,2015-2016,0.72,0.18,62000,1850
```

### provenance.jsonl

Provenance records for audit trail:

```json
{"artifact_id": "mit_catalog:6.006", "source_url": "...", "source_hash": "...", "timestamp": "..."}
```

## Cross-Dataset Comparison

### Comparable Metrics

| Metric | Description | Computed From |
|--------|-------------|---------------|
| `artifact_count` | Number of artifacts | `len(artifacts)` |
| `pareto_count` | Pareto-optimal options | Evaluator output |
| `pareto_ratio` | Pareto/total ratio | `pareto_count / artifact_count` |
| `mean_learner_score` | Average learner objective | Evaluator output |
| `mean_market_score` | Average market objective | Evaluator output |

### Normalization for Comparison

When comparing across institutions:

1. **Embedding**: Use same model for all datasets
2. **Lenses**: Apply identical lens configuration
3. **Constraints**: Same autonomy drift tau
4. **Learner profile**: Same reference text

## License Summary

| Institution | Data Type | License |
|-------------|-----------|---------|
| MIT | Course Catalog | Public information |
| MIT | OCW Materials | CC BY-NC-SA 4.0 |
| UCB | FDS Outcomes | Public aggregate data |
| ASU | Career Outcomes | Public report |
| ASU | FDS Survey | Public document |
| O*NET | Job/Skill Data | Public domain |

## Validation Requirements

Each artifact must pass:

1. **Required fields present**: `artifact_id`, `type`, `text`
2. **Institution prefix**: ID starts with valid prefix
3. **Provenance complete**: `source_url`, `fetched_at`, `raw_text_hash`
4. **License specified**: Non-empty license field
5. **Text non-empty**: `len(text.strip()) > 0`

## Usage Example

```python
from ads_core.datasets import load_dataset

# Load MIT dataset
mit = load_dataset("mit")
print(f"MIT: {len(mit.artifacts)} artifacts")

# Load UCB dataset
ucb = load_dataset("ucb")
print(f"UCB: {len(ucb.artifacts)} artifacts, {len(ucb.outcomes)} outcomes")

# Compare datasets
for ds in [mit, ucb, asu]:
    print(f"{ds.dataset_id}: {ds.metadata['artifact_count']} artifacts")
```
