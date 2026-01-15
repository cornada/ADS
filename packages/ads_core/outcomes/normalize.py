"""Outcomes normalization for cross-dataset comparison.

Normalizes UCB and ASU outcomes data into a canonical schema for
comparing predicted market-fit with observed career outcomes.

Canonical Schema:
- institution: MIT, UCB, ASU
- unit: College/department code
- unit_name: Full name
- cohort_year: Academic year (e.g., "2024", "2015-2016")
- employment_rate: Float [0, 1]
- grad_school_rate: Float [0, 1]
- career_outcomes_rate: Float [0, 1] (employed + grad school)
- seeking_rate: Float [0, 1]
- median_salary: Float (USD)
- n_respondents: Int
- source_url: Provenance URL
- raw_hash: SHA-256 of source data
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class CanonicalOutcome:
    """Canonical outcome record for cross-dataset comparison."""

    institution: str
    unit: str
    unit_name: str
    cohort_year: str
    employment_rate: Optional[float] = None
    grad_school_rate: Optional[float] = None
    career_outcomes_rate: Optional[float] = None
    seeking_rate: Optional[float] = None
    median_salary: Optional[float] = None
    n_respondents: Optional[int] = None
    source_url: str = ""
    raw_hash: str = ""

    def __post_init__(self):
        """Compute derived fields if not set."""
        # Compute career_outcomes_rate if we have employment and grad school
        if self.career_outcomes_rate is None:
            if self.employment_rate is not None and self.grad_school_rate is not None:
                self.career_outcomes_rate = min(1.0, self.employment_rate + self.grad_school_rate)

        # Compute seeking_rate if we have career_outcomes
        if self.seeking_rate is None and self.career_outcomes_rate is not None:
            self.seeking_rate = max(0.0, 1.0 - self.career_outcomes_rate)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "institution": self.institution,
            "unit": self.unit,
            "unit_name": self.unit_name,
            "cohort_year": self.cohort_year,
            "employment_rate": self.employment_rate,
            "grad_school_rate": self.grad_school_rate,
            "career_outcomes_rate": self.career_outcomes_rate,
            "seeking_rate": self.seeking_rate,
            "median_salary": self.median_salary,
            "n_respondents": self.n_respondents,
            "source_url": self.source_url,
            "raw_hash": self.raw_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanonicalOutcome":
        """Create from dictionary."""
        return cls(
            institution=d.get("institution", ""),
            unit=d.get("unit", ""),
            unit_name=d.get("unit_name", ""),
            cohort_year=str(d.get("cohort_year", "")),
            employment_rate=d.get("employment_rate"),
            grad_school_rate=d.get("grad_school_rate"),
            career_outcomes_rate=d.get("career_outcomes_rate"),
            seeking_rate=d.get("seeking_rate"),
            median_salary=d.get("median_salary"),
            n_respondents=d.get("n_respondents"),
            source_url=d.get("source_url", ""),
            raw_hash=d.get("raw_hash", ""),
        )


# UCB college name mappings
UCB_COLLEGE_NAMES = {
    "COE": "College of Engineering",
    "L&S": "College of Letters & Science",
    "HAAS": "Haas School of Business",
    "COC": "College of Chemistry",
    "CNR": "College of Natural Resources",
    "CED": "College of Environmental Design",
    "CDPH": "School of Public Health",
    "EDU": "Graduate School of Education",
}


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def normalize_ucb_outcomes(
    csv_path: Path,
    source_url: str = "https://career.berkeley.edu/start-exploring/where-do-cal-grads-go/",
) -> List[CanonicalOutcome]:
    """Normalize UC Berkeley FDS outcomes to canonical format.

    Args:
        csv_path: Path to UCB outcomes CSV
        source_url: Provenance URL

    Returns:
        List of CanonicalOutcome records
    """
    records = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        content = f.read()
        raw_hash = _compute_hash(content)

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            year = row.get("year", "")
            college = row.get("college", "").strip()
            major = row.get("major", "").strip()

            if not college or not major:
                continue

            # Use major as the unit for UCB (more granular than college)
            unit = f"{college}:{major}"
            unit_name = f"{major} ({UCB_COLLEGE_NAMES.get(college, college)})"

            record = CanonicalOutcome(
                institution="UCB",
                unit=unit,
                unit_name=unit_name,
                cohort_year=str(year),
                employment_rate=_parse_rate(row.get("employment_rate")),
                grad_school_rate=_parse_rate(row.get("grad_school_rate")),
                median_salary=_parse_salary(row.get("median_salary")),
                n_respondents=_parse_int(row.get("n_respondents")),
                source_url=source_url,
                raw_hash=raw_hash,
            )
            records.append(record)

    return records


def normalize_asu_outcomes(
    csv_path: Path,
    source_url: str = "https://provost.asu.edu/career-outcomes",
) -> List[CanonicalOutcome]:
    """Normalize ASU career outcomes to canonical format.

    Args:
        csv_path: Path to ASU outcomes CSV
        source_url: Provenance URL

    Returns:
        List of CanonicalOutcome records
    """
    records = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        content = f.read()
        raw_hash = _compute_hash(content)

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            college = row.get("college", "").strip()
            college_name = row.get("college_name", "").strip()
            academic_year = row.get("academic_year", "").strip()

            if not college:
                continue

            # Skip the overall ASU row for unit-level analysis
            # (keep it but mark it differently)
            unit = college
            unit_name = college_name or college

            record = CanonicalOutcome(
                institution="ASU",
                unit=unit,
                unit_name=unit_name,
                cohort_year=academic_year,
                employment_rate=_parse_rate(row.get("employment_rate")),
                grad_school_rate=_parse_rate(row.get("grad_school_rate")),
                career_outcomes_rate=_parse_rate(row.get("career_outcomes_rate")),
                seeking_rate=_parse_rate(row.get("seeking_rate")),
                median_salary=_parse_salary(row.get("median_salary")),
                n_respondents=_parse_int(row.get("n_respondents")),
                source_url=source_url,
                raw_hash=raw_hash,
            )
            records.append(record)

    return records


def _parse_rate(value: Optional[str]) -> Optional[float]:
    """Parse a rate value, handling percentages."""
    if value is None or not str(value).strip():
        return None

    value = str(value).strip()
    if value in ("-", "N/A", "n/a", ""):
        return None

    # Handle percentage format
    if value.endswith("%"):
        try:
            return float(value[:-1]) / 100.0
        except ValueError:
            return None

    try:
        val = float(value)
        # If > 1, assume it's a percentage without %
        if val > 1:
            return val / 100.0
        return val
    except ValueError:
        return None


def _parse_salary(value: Optional[str]) -> Optional[float]:
    """Parse a salary value."""
    if value is None or not str(value).strip():
        return None

    value = str(value).strip()
    if value in ("-", "N/A", "n/a", ""):
        return None

    # Remove $ and commas
    value = value.replace("$", "").replace(",", "")

    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    """Parse an integer value."""
    if value is None or not str(value).strip():
        return None

    value = str(value).strip()
    if value in ("-", "N/A", "n/a", ""):
        return None

    value = value.replace(",", "")

    try:
        return int(float(value))
    except ValueError:
        return None


def load_normalized_outcomes(
    dataset_id: str,
    data_dir: Optional[Path] = None,
) -> List[CanonicalOutcome]:
    """Load normalized outcomes for a dataset.

    Args:
        dataset_id: Dataset ID (ucb, asu)
        data_dir: Base data directory

    Returns:
        List of CanonicalOutcome records
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent.parent

    if dataset_id == "ucb":
        # Try processed first, then fixtures
        processed_path = data_dir / "data" / "processed" / "ucb" / "outcomes_major_level.csv"
        fixture_path = data_dir / "data" / "fixtures" / "ucb" / "fds_outcomes_by_major.csv"

        if processed_path.exists():
            return normalize_ucb_outcomes(processed_path)
        elif fixture_path.exists():
            return normalize_ucb_outcomes(fixture_path)
        else:
            raise FileNotFoundError(f"No UCB outcomes found at {processed_path} or {fixture_path}")

    elif dataset_id == "asu":
        processed_path = data_dir / "data" / "processed" / "asu" / "outcomes_summary.csv"
        fixture_path = data_dir / "data" / "fixtures" / "asu" / "outcomes_summary.csv"

        if processed_path.exists():
            return normalize_asu_outcomes(processed_path)
        elif fixture_path.exists():
            return normalize_asu_outcomes(fixture_path)
        else:
            raise FileNotFoundError(f"No ASU outcomes found at {processed_path} or {fixture_path}")

    else:
        raise ValueError(f"No outcomes available for dataset: {dataset_id}")


def outcomes_to_dataframe(outcomes: List[CanonicalOutcome]) -> pd.DataFrame:
    """Convert outcomes list to DataFrame."""
    return pd.DataFrame([o.to_dict() for o in outcomes])


def export_normalized_outcomes(
    outcomes: List[CanonicalOutcome],
    output_path: Path,
) -> None:
    """Export normalized outcomes to CSV.

    Args:
        outcomes: List of CanonicalOutcome records
        output_path: Output CSV path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "institution", "unit", "unit_name", "cohort_year",
        "employment_rate", "grad_school_rate", "career_outcomes_rate",
        "seeking_rate", "median_salary", "n_respondents",
        "source_url", "raw_hash",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for o in outcomes:
            writer.writerow(o.to_dict())
