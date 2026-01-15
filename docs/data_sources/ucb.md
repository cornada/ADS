# UC Berkeley Outcomes Data Source

## Overview

This document describes the UC Berkeley career outcomes data source used in ADS for validating market-fit predictions against real employment outcomes.

## Data Sources

### First Destination Survey (FDS)
- **URL**: https://opa.berkeley.edu/campus-surveys/survey-results-reporting-analysis/first-destination-survey
- **Provider**: UC Berkeley Office of Planning and Analysis (OPA)
- **Type**: Aggregate career outcomes by major/college
- **License**: Public aggregate data for educational research

### Where Do Cal Grads Go?
- **URL**: https://career.berkeley.edu/start-exploring/where-do-cal-grads-go/
- **Provider**: UC Berkeley Career Engagement
- **Type**: Career outcomes dashboard
- **License**: Public information

## Data Schema

The normalized output (`outcomes_major_level.csv`) contains:

| Column | Type | Description |
|--------|------|-------------|
| `year` | int | Survey/graduation year |
| `college` | str | College code (COE, L&S, HAAS, etc.) |
| `major` | str | Degree program name |
| `employment_rate` | float | Employment rate (0-1) |
| `grad_school_rate` | float | Graduate school rate (0-1) |
| `median_salary` | float | Median starting salary (USD) |
| `n_respondents` | int | Number of survey respondents |
| `response_rate` | float | Survey response rate (0-1) |

### College Codes

| Code | Name |
|------|------|
| CNR | Rausser College of Natural Resources |
| CED | College of Environmental Design |
| COC | College of Chemistry |
| COE | College of Engineering |
| HAAS | Haas School of Business |
| L&S | College of Letters and Science |

## Usage

### Basic Ingestion

```bash
# Using fixtures (reproducible, default)
python -m ads_core.ingest.ucb --manifest data/manifests/ucb_sources.yaml --out_dir data/processed/ucb -v

# View generated outcomes
python -c "import pandas as pd; df=pd.read_csv('data/processed/ucb/outcomes_major_level.csv'); print(df.head(10))"
```

### Custom Data Directory

```bash
# Using manually downloaded CSV files
python -m ads_core.ingest.ucb \
  --manifest data/manifests/ucb_sources.yaml \
  --raw_dir data/raw/ucb/fds \
  --out_dir data/processed/ucb
```

### Creating Fixtures

```bash
python -m ads_core.ingest.ucb --create-fixtures
```

## Manual Export Procedure

Since the FDS dashboard uses Tableau (JS-heavy), direct scraping is not supported. Use this procedure to create reproducible snapshots:

### Step 1: Export from FDS Dashboard

1. Navigate to the FDS results page
2. Use Tableau's export functionality to download CSV
3. Save as `data/raw/ucb/fds/fds_outcomes_by_major.csv`

### Step 2: Record Provenance

Create `data/raw/ucb/fds/provenance.json`:

```json
{
  "source_url": "https://opa.berkeley.edu/...",
  "retrieved_at": "2026-01-15T12:00:00Z",
  "export_method": "Tableau CSV export",
  "notes": "Manual export from FDS dashboard"
}
```

### Step 3: Compute Hash

```bash
# Record hash for reproducibility
python -c "from ads_core.utils.hashing import sha256_text; print(sha256_text(open('data/raw/ucb/fds/fds_outcomes_by_major.csv').read()))"
```

### Step 4: Run Ingestion

```bash
python -m ads_core.ingest.ucb \
  --manifest data/manifests/ucb_sources.yaml \
  --raw_dir data/raw/ucb/fds \
  --out_dir data/processed/ucb -v
```

## CSV Format Requirements

The ingestion supports flexible column naming. Accepted column names:

| Field | Accepted Names |
|-------|---------------|
| Year | `year`, `survey_year`, `academic_year`, `grad_year` |
| College | `college`, `school`, `college_code`, `college_name` |
| Major | `major`, `major_name`, `program`, `degree_program` |
| Employment Rate | `employment_rate`, `employed_pct`, `emp_rate` |
| Grad School Rate | `grad_school_rate`, `grad_school_pct`, `grad_rate` |
| Median Salary | `median_salary`, `salary_median`, `salary` |
| N Respondents | `n_respondents`, `respondents`, `n`, `sample_size` |

Values can be:
- Decimals: `0.85`
- Percentages: `85%`
- Currency: `$75,000`
- Empty: `-`, `N/A`, `n/a`, blank

## Output Files

| File | Description |
|------|-------------|
| `outcomes_major_level.csv` | Normalized outcomes by major |
| `artifacts.jsonl` | Text artifacts for embedding |
| `provenance.jsonl` | Provenance records |
| `manifest_snapshot.yaml` | Ingestion run metadata |

## Artifact Format

Each outcome becomes an `OUTCOME_MAJOR` artifact:

```json
{
  "artifact_id": "ucb_outcome:2024:COE:Computer_Science",
  "type": "OUTCOME_MAJOR",
  "text": "UC Berkeley Computer Science graduates from COE (2024): Employment rate 85.0%. Graduate school rate 12.0%. Median starting salary $130,000. Based on 450 respondents.",
  "metadata": {
    "year": 2024,
    "college": "COE",
    "major": "Computer Science",
    "employment_rate": 0.85,
    "grad_school_rate": 0.12,
    "median_salary": 130000,
    "n_respondents": 450,
    "license": "Public aggregate data for educational research"
  }
}
```

## Validation Use Cases

The UCB outcomes data enables:

1. **Market-fit validation**: Compare embedding-based predictions to actual employment rates
2. **Salary benchmarking**: Validate salary predictions against median outcomes
3. **Program comparison**: Analyze trade-offs between majors
4. **Trend analysis**: Track year-over-year changes in outcomes

## Privacy Considerations

- Data is aggregate only (no individual-level outcomes)
- All data is publicly available from UC Berkeley
- No PII is collected or stored
