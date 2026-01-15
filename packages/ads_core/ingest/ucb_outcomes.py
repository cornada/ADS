"""UC Berkeley career outcomes ingestion connector.

Ingests First Destination Survey (FDS) data and career outcomes from UC Berkeley.
Designed to work with manually exported CSV snapshots since official dashboards
are JS-heavy (Tableau) and not directly scrapeable.

Data sources:
- First Destination Survey: https://opa.berkeley.edu/campus-surveys/survey-results-reporting-analysis/first-destination-survey
- Where Do Cal Grads Go?: https://career.berkeley.edu/start-exploring/where-do-cal-grads-go/

License: Public aggregate data for educational research
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml

from ads_core.data.schemas import Artifact, ArtifactType
from ads_core.ingest.base import BaseConnector, create_provenance


# Column mappings for flexible CSV parsing
COLUMN_ALIASES = {
    "year": ["year", "survey_year", "academic_year", "grad_year"],
    "college": ["college", "school", "college_code", "college_name"],
    "major": ["major", "major_name", "program", "degree_program"],
    "employment_rate": ["employment_rate", "employed_pct", "employment_pct", "emp_rate"],
    "grad_school_rate": ["grad_school_rate", "grad_school_pct", "continuing_education_pct", "grad_rate"],
    "median_salary": ["median_salary", "salary_median", "median_annual_salary", "salary"],
    "n_respondents": ["n_respondents", "respondents", "n", "sample_size", "count"],
    "response_rate": ["response_rate", "response_pct"],
}


@dataclass
class OutcomeRecord:
    """Parsed outcome record from FDS data."""
    year: int
    college: str
    major: str
    employment_rate: Optional[float] = None
    grad_school_rate: Optional[float] = None
    median_salary: Optional[float] = None
    n_respondents: Optional[int] = None
    response_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        d = {
            "year": self.year,
            "college": self.college,
            "major": self.major,
        }
        if self.employment_rate is not None:
            d["employment_rate"] = self.employment_rate
        if self.grad_school_rate is not None:
            d["grad_school_rate"] = self.grad_school_rate
        if self.median_salary is not None:
            d["median_salary"] = self.median_salary
        if self.n_respondents is not None:
            d["n_respondents"] = self.n_respondents
        if self.response_rate is not None:
            d["response_rate"] = self.response_rate
        return d

    def to_text(self) -> str:
        """Convert to text description for embedding."""
        parts = [f"UC Berkeley {self.major} graduates from {self.college} ({self.year}):"]

        if self.employment_rate is not None:
            parts.append(f"Employment rate {self.employment_rate:.1%}.")
        if self.grad_school_rate is not None:
            parts.append(f"Graduate school rate {self.grad_school_rate:.1%}.")
        if self.median_salary is not None:
            parts.append(f"Median starting salary ${self.median_salary:,.0f}.")
        if self.n_respondents is not None:
            parts.append(f"Based on {self.n_respondents} respondents.")

        return " ".join(parts)


def find_column(headers: List[str], field: str) -> Optional[int]:
    """Find column index for a field, trying aliases."""
    aliases = COLUMN_ALIASES.get(field, [field])
    headers_lower = [h.lower().strip() for h in headers]

    for alias in aliases:
        if alias.lower() in headers_lower:
            return headers_lower.index(alias.lower())
    return None


def parse_float(value: str) -> Optional[float]:
    """Parse a float value, handling percentages and empty strings."""
    if not value or value.strip() in ("", "-", "N/A", "n/a", "NA"):
        return None

    value = value.strip()

    # Handle percentage format
    if value.endswith("%"):
        try:
            return float(value[:-1]) / 100.0
        except ValueError:
            return None

    # Handle currency format
    value = value.replace("$", "").replace(",", "")

    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str) -> Optional[int]:
    """Parse an integer value."""
    if not value or value.strip() in ("", "-", "N/A", "n/a", "NA"):
        return None

    value = value.strip().replace(",", "")

    try:
        return int(float(value))  # Handle "123.0" format
    except ValueError:
        return None


def parse_outcomes_csv(csv_path: Path) -> List[OutcomeRecord]:
    """Parse outcomes data from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of OutcomeRecord objects
    """
    records = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:  # utf-8-sig handles BOM
        reader = csv.reader(f)
        headers = next(reader)

        # Find column indices
        year_idx = find_column(headers, "year")
        college_idx = find_column(headers, "college")
        major_idx = find_column(headers, "major")
        emp_idx = find_column(headers, "employment_rate")
        grad_idx = find_column(headers, "grad_school_rate")
        salary_idx = find_column(headers, "median_salary")
        n_idx = find_column(headers, "n_respondents")
        resp_idx = find_column(headers, "response_rate")

        if major_idx is None:
            raise ValueError(f"CSV must have a 'major' column. Found: {headers}")

        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue  # Skip empty rows

            # Parse required fields
            year = parse_int(row[year_idx]) if year_idx is not None else 2024
            college = row[college_idx].strip() if college_idx is not None else "Unknown"
            major = row[major_idx].strip() if major_idx is not None else ""

            if not major:
                continue

            record = OutcomeRecord(
                year=year or 2024,
                college=college,
                major=major,
                employment_rate=parse_float(row[emp_idx]) if emp_idx is not None and emp_idx < len(row) else None,
                grad_school_rate=parse_float(row[grad_idx]) if grad_idx is not None and grad_idx < len(row) else None,
                median_salary=parse_float(row[salary_idx]) if salary_idx is not None and salary_idx < len(row) else None,
                n_respondents=parse_int(row[n_idx]) if n_idx is not None and n_idx < len(row) else None,
                response_rate=parse_float(row[resp_idx]) if resp_idx is not None and resp_idx < len(row) else None,
            )
            records.append(record)

    return records


class UcbOutcomesConnector(BaseConnector):
    """Connector for UC Berkeley career outcomes data.

    Designed to work with manually exported CSV snapshots since the official
    dashboards are JS-heavy (Tableau) and not directly scrapeable.

    Supports:
    - CSV files exported from FDS dashboards
    - Fixture data for testing
    - Provenance tracking with file hashes
    """

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
        max_records: Optional[int] = None,
    ):
        """Initialize UCB outcomes connector.

        Args:
            manifest_path: Path to ucb_sources.yaml manifest
            raw_dir: Directory containing raw CSV files
            max_records: Limit records (for testing)
        """
        self.manifest_path = manifest_path
        self.raw_dir = raw_dir
        self.max_records = max_records
        self._manifest: Optional[Dict] = None
        self._records: Optional[List[OutcomeRecord]] = None

    @property
    def source_name(self) -> str:
        return "ucb_outcomes"

    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest configuration."""
        if self._manifest is not None:
            return self._manifest

        if self.manifest_path and Path(self.manifest_path).exists():
            self._manifest = yaml.safe_load(
                Path(self.manifest_path).read_text(encoding="utf-8")
            )
        else:
            self._manifest = {
                "version": "1.0",
                "institution": "ucb",
                "settings": {"use_fixtures": True},
            }

        return self._manifest

    def _find_csv_files(self) -> List[Path]:
        """Find CSV files to process."""
        csv_files = []

        # Check raw_dir first
        if self.raw_dir and Path(self.raw_dir).exists():
            csv_files.extend(Path(self.raw_dir).glob("*.csv"))

        # Fall back to fixtures
        if not csv_files:
            fixture_dir = Path(__file__).parent.parent.parent.parent / "data" / "fixtures" / "ucb"
            if fixture_dir.exists():
                csv_files.extend(fixture_dir.glob("*.csv"))

        return sorted(csv_files)

    def load_records(self) -> List[OutcomeRecord]:
        """Load all outcome records from CSV files."""
        if self._records is not None:
            return self._records

        self._records = []

        for csv_path in self._find_csv_files():
            try:
                records = parse_outcomes_csv(csv_path)
                self._records.extend(records)
            except Exception as e:
                print(f"Warning: Could not parse {csv_path}: {e}")

        return self._records

    def fetch(self) -> Iterator[Artifact]:
        """Fetch outcome artifacts."""
        records = self.load_records()
        manifest = self._load_manifest()

        source_url = manifest.get("sources", {}).get("fds_overview", {}).get(
            "url", "https://opa.berkeley.edu/campus-surveys/survey-results-reporting-analysis/first-destination-survey"
        )

        count = 0
        for record in records:
            if self.max_records and count >= self.max_records:
                break

            text = record.to_text()
            if not text.strip():
                continue

            # Create artifact ID
            safe_major = record.major.replace(" ", "_").replace("/", "_")[:50]
            artifact_id = f"ucb_outcome:{record.year}:{record.college}:{safe_major}"

            prov = create_provenance(
                source_url=source_url,
                raw_text=text,
                year=record.year,
                college=record.college,
                major=record.major,
            )

            yield Artifact(
                artifact_id=artifact_id,
                type=ArtifactType.OUTCOME_MAJOR,
                source_url=source_url,
                source_hash=prov.raw_text_hash,
                timestamp=prov.fetched_at,
                text=text,
                metadata={
                    **record.to_dict(),
                    "license": "Public aggregate data for educational research",
                    "provenance": prov.to_dict(),
                },
            )
            count += 1


def export_outcomes_csv(records: List[OutcomeRecord], output_path: Path) -> int:
    """Export outcome records to normalized CSV.

    Args:
        records: List of OutcomeRecord objects
        output_path: Path for output CSV

    Returns:
        Number of records written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "year", "college", "major",
        "employment_rate", "grad_school_rate",
        "median_salary", "n_respondents", "response_rate"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            row = record.to_dict()
            # Format percentages as decimals
            if row.get("employment_rate") is not None:
                row["employment_rate"] = round(row["employment_rate"], 4)
            if row.get("grad_school_rate") is not None:
                row["grad_school_rate"] = round(row["grad_school_rate"], 4)
            if row.get("response_rate") is not None:
                row["response_rate"] = round(row["response_rate"], 4)
            writer.writerow(row)

    return len(records)


def create_ucb_fixture(output_dir: Path) -> None:
    """Create a minimal UCB fixture for testing.

    Creates sample FDS-like data based on publicly available aggregate statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample outcomes data (representative of FDS aggregates)
    # Based on publicly available Berkeley outcomes statistics
    sample_data = [
        # College of Engineering
        {"year": 2024, "college": "COE", "major": "Computer Science", "employment_rate": 0.85, "grad_school_rate": 0.12, "median_salary": 130000, "n_respondents": 450},
        {"year": 2024, "college": "COE", "major": "Electrical Engineering", "employment_rate": 0.82, "grad_school_rate": 0.15, "median_salary": 115000, "n_respondents": 180},
        {"year": 2024, "college": "COE", "major": "Mechanical Engineering", "employment_rate": 0.80, "grad_school_rate": 0.14, "median_salary": 95000, "n_respondents": 200},
        {"year": 2024, "college": "COE", "major": "Bioengineering", "employment_rate": 0.70, "grad_school_rate": 0.25, "median_salary": 85000, "n_respondents": 120},
        {"year": 2024, "college": "COE", "major": "Civil Engineering", "employment_rate": 0.78, "grad_school_rate": 0.10, "median_salary": 80000, "n_respondents": 90},

        # College of Letters and Science
        {"year": 2024, "college": "L&S", "major": "Data Science", "employment_rate": 0.83, "grad_school_rate": 0.10, "median_salary": 110000, "n_respondents": 300},
        {"year": 2024, "college": "L&S", "major": "Economics", "employment_rate": 0.78, "grad_school_rate": 0.15, "median_salary": 75000, "n_respondents": 350},
        {"year": 2024, "college": "L&S", "major": "Statistics", "employment_rate": 0.82, "grad_school_rate": 0.12, "median_salary": 95000, "n_respondents": 150},
        {"year": 2024, "college": "L&S", "major": "Mathematics", "employment_rate": 0.65, "grad_school_rate": 0.30, "median_salary": 85000, "n_respondents": 120},
        {"year": 2024, "college": "L&S", "major": "Psychology", "employment_rate": 0.60, "grad_school_rate": 0.25, "median_salary": 55000, "n_respondents": 280},
        {"year": 2024, "college": "L&S", "major": "Political Science", "employment_rate": 0.55, "grad_school_rate": 0.30, "median_salary": 52000, "n_respondents": 200},
        {"year": 2024, "college": "L&S", "major": "English", "employment_rate": 0.50, "grad_school_rate": 0.20, "median_salary": 48000, "n_respondents": 100},

        # Haas School of Business
        {"year": 2024, "college": "HAAS", "major": "Business Administration", "employment_rate": 0.90, "grad_school_rate": 0.05, "median_salary": 95000, "n_respondents": 350},

        # College of Chemistry
        {"year": 2024, "college": "COC", "major": "Chemistry", "employment_rate": 0.55, "grad_school_rate": 0.40, "median_salary": 65000, "n_respondents": 80},
        {"year": 2024, "college": "COC", "major": "Chemical Biology", "employment_rate": 0.50, "grad_school_rate": 0.45, "median_salary": 60000, "n_respondents": 60},

        # College of Natural Resources
        {"year": 2024, "college": "CNR", "major": "Environmental Science", "employment_rate": 0.65, "grad_school_rate": 0.20, "median_salary": 55000, "n_respondents": 90},
        {"year": 2024, "college": "CNR", "major": "Nutritional Science", "employment_rate": 0.60, "grad_school_rate": 0.30, "median_salary": 50000, "n_respondents": 70},

        # College of Environmental Design
        {"year": 2024, "college": "CED", "major": "Architecture", "employment_rate": 0.70, "grad_school_rate": 0.20, "median_salary": 60000, "n_respondents": 80},

        # Previous year data
        {"year": 2023, "college": "COE", "major": "Computer Science", "employment_rate": 0.88, "grad_school_rate": 0.10, "median_salary": 125000, "n_respondents": 420},
        {"year": 2023, "college": "L&S", "major": "Data Science", "employment_rate": 0.85, "grad_school_rate": 0.08, "median_salary": 105000, "n_respondents": 250},
    ]

    # Write fixture CSV
    fieldnames = ["year", "college", "major", "employment_rate", "grad_school_rate", "median_salary", "n_respondents"]

    with open(output_dir / "fds_outcomes_by_major.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample_data:
            writer.writerow(row)

    # Write metadata
    meta = {
        "source": "UC Berkeley FDS - synthetic fixture for testing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "license": "Public aggregate data for educational research",
        "notes": "Sample data representative of FDS aggregate statistics. Not actual survey data.",
        "original_url": "https://opa.berkeley.edu/campus-surveys/survey-results-reporting-analysis/first-destination-survey",
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
