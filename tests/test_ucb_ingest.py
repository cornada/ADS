"""Tests for UC Berkeley outcomes ingestion."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from ads_core.data.schemas import ArtifactType
from ads_core.ingest.ucb_outcomes import (
    UcbOutcomesConnector,
    OutcomeRecord,
    parse_outcomes_csv,
    export_outcomes_csv,
    create_ucb_fixture,
    parse_float,
    parse_int,
    find_column,
)


class TestParsing:
    """Test CSV parsing utilities."""

    def test_parse_float_decimal(self):
        """Test parsing decimal float."""
        assert parse_float("0.85") == 0.85
        assert parse_float("0.123") == 0.123

    def test_parse_float_percentage(self):
        """Test parsing percentage format."""
        assert parse_float("85%") == 0.85
        assert parse_float("12.5%") == 0.125

    def test_parse_float_currency(self):
        """Test parsing currency format."""
        assert parse_float("$75,000") == 75000.0
        assert parse_float("$1,234,567") == 1234567.0

    def test_parse_float_empty(self):
        """Test parsing empty values."""
        assert parse_float("") is None
        assert parse_float("-") is None
        assert parse_float("N/A") is None
        assert parse_float("n/a") is None

    def test_parse_int(self):
        """Test integer parsing."""
        assert parse_int("100") == 100
        assert parse_int("1,234") == 1234
        assert parse_int("100.0") == 100

    def test_parse_int_empty(self):
        """Test parsing empty integer values."""
        assert parse_int("") is None
        assert parse_int("-") is None

    def test_find_column(self):
        """Test column finding with aliases."""
        headers = ["Year", "Major", "Employment_Rate"]
        assert find_column(headers, "year") == 0
        assert find_column(headers, "major") == 1
        assert find_column(headers, "employment_rate") == 2

    def test_find_column_aliases(self):
        """Test column aliases."""
        headers = ["survey_year", "program", "emp_rate"]
        assert find_column(headers, "year") == 0
        assert find_column(headers, "major") == 1
        assert find_column(headers, "employment_rate") == 2

    def test_find_column_missing(self):
        """Test missing column."""
        headers = ["a", "b", "c"]
        assert find_column(headers, "nonexistent") is None


class TestOutcomeRecord:
    """Test OutcomeRecord dataclass."""

    def test_to_dict(self):
        """Test conversion to dict."""
        record = OutcomeRecord(
            year=2024,
            college="COE",
            major="Computer Science",
            employment_rate=0.85,
            median_salary=130000,
        )
        d = record.to_dict()
        assert d["year"] == 2024
        assert d["college"] == "COE"
        assert d["major"] == "Computer Science"
        assert d["employment_rate"] == 0.85
        assert d["median_salary"] == 130000
        assert "grad_school_rate" not in d  # None values excluded

    def test_to_text(self):
        """Test conversion to text."""
        record = OutcomeRecord(
            year=2024,
            college="COE",
            major="Computer Science",
            employment_rate=0.85,
            grad_school_rate=0.12,
            median_salary=130000,
            n_respondents=450,
        )
        text = record.to_text()
        assert "Computer Science" in text
        assert "COE" in text
        assert "2024" in text
        assert "85.0%" in text
        assert "12.0%" in text
        assert "$130,000" in text
        assert "450 respondents" in text


class TestCsvParsing:
    """Test CSV file parsing."""

    def test_parse_basic_csv(self):
        """Test parsing basic CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text(
                "year,college,major,employment_rate,grad_school_rate,median_salary,n_respondents\n"
                "2024,COE,Computer Science,0.85,0.12,130000,450\n"
                "2024,L&S,Economics,0.78,0.15,75000,350\n",
                encoding="utf-8"
            )

            records = parse_outcomes_csv(csv_path)
            assert len(records) == 2

            cs = records[0]
            assert cs.year == 2024
            assert cs.college == "COE"
            assert cs.major == "Computer Science"
            assert cs.employment_rate == 0.85
            assert cs.grad_school_rate == 0.12
            assert cs.median_salary == 130000
            assert cs.n_respondents == 450

    def test_parse_percentage_format(self):
        """Test parsing percentages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text(
                "major,employment_rate,college\n"
                "CS,85%,COE\n",
                encoding="utf-8"
            )

            records = parse_outcomes_csv(csv_path)
            assert len(records) == 1
            assert records[0].employment_rate == 0.85

    def test_parse_with_missing_values(self):
        """Test parsing with missing values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text(
                "year,college,major,employment_rate,median_salary\n"
                "2024,COE,CS,0.85,-\n"
                "2024,L&S,Econ,N/A,75000\n",
                encoding="utf-8"
            )

            records = parse_outcomes_csv(csv_path)
            assert len(records) == 2
            assert records[0].median_salary is None
            assert records[1].employment_rate is None

    def test_parse_skip_empty_rows(self):
        """Test skipping empty rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text(
                "year,college,major,employment_rate\n"
                "2024,COE,CS,0.85\n"
                "\n"
                "2024,L&S,Econ,0.78\n",
                encoding="utf-8"
            )

            records = parse_outcomes_csv(csv_path)
            assert len(records) == 2


class TestUcbOutcomesConnector:
    """Test UCB outcomes connector."""

    def test_connector_with_fixture(self):
        """Test connector using created fixture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir) / "fixtures"
            create_ucb_fixture(fixture_dir)

            connector = UcbOutcomesConnector(raw_dir=fixture_dir)

            artifacts = list(connector.fetch())
            assert len(artifacts) > 0

            # Check artifact structure
            art = artifacts[0]
            assert art.artifact_id.startswith("ucb_outcome:")
            assert art.type == ArtifactType.OUTCOME_MAJOR
            assert "year" in art.metadata
            assert "college" in art.metadata
            assert "major" in art.metadata
            assert "provenance" in art.metadata

    def test_connector_max_records(self):
        """Test max_records limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir) / "fixtures"
            create_ucb_fixture(fixture_dir)

            connector = UcbOutcomesConnector(
                raw_dir=fixture_dir,
                max_records=5,
            )

            artifacts = list(connector.fetch())
            assert len(artifacts) == 5

    def test_connector_source_name(self):
        """Test connector source name."""
        connector = UcbOutcomesConnector()
        assert connector.source_name == "ucb_outcomes"

    def test_load_records(self):
        """Test loading records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir) / "fixtures"
            create_ucb_fixture(fixture_dir)

            connector = UcbOutcomesConnector(raw_dir=fixture_dir)
            records = connector.load_records()

            assert len(records) > 0
            assert all(isinstance(r, OutcomeRecord) for r in records)


class TestFixtureCreation:
    """Test fixture creation."""

    def test_create_ucb_fixture(self):
        """Test UCB fixture creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir)
            create_ucb_fixture(fixture_dir)

            assert (fixture_dir / "fds_outcomes_by_major.csv").exists()
            assert (fixture_dir / "meta.json").exists()

            # Check CSV content
            with open(fixture_dir / "fds_outcomes_by_major.csv", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) > 0
            assert "major" in rows[0]
            assert "employment_rate" in rows[0]

            # Check for Computer Science (highest employment)
            cs_rows = [r for r in rows if r["major"] == "Computer Science"]
            assert len(cs_rows) > 0


class TestExportCsv:
    """Test CSV export functionality."""

    def test_export_outcomes_csv(self):
        """Test exporting outcomes to CSV."""
        records = [
            OutcomeRecord(
                year=2024,
                college="COE",
                major="Computer Science",
                employment_rate=0.85,
                median_salary=130000,
            ),
            OutcomeRecord(
                year=2024,
                college="L&S",
                major="Economics",
                employment_rate=0.78,
                median_salary=75000,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "outcomes.csv"
            count = export_outcomes_csv(records, output_path)

            assert count == 2
            assert output_path.exists()

            # Verify content
            with open(output_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["major"] == "Computer Science"
            assert float(rows[0]["employment_rate"]) == 0.85


class TestIntegration:
    """Integration tests for UCB ingestion."""

    def test_full_ingestion_flow(self):
        """Test complete ingestion from fixtures to artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fixture_dir = tmpdir / "fixtures"
            output_dir = tmpdir / "output"

            # Create fixtures
            create_ucb_fixture(fixture_dir)

            # Create manifest
            manifest_path = tmpdir / "manifest.yaml"
            manifest = {
                "version": "1.0",
                "institution": "ucb",
                "sources": {
                    "fds_overview": {
                        "url": "https://opa.berkeley.edu/fds"
                    }
                },
            }
            manifest_path.write_text(
                __import__("yaml").dump(manifest),
                encoding="utf-8"
            )

            # Run ingestion
            from ads_core.ingest.ucb import run_ingestion

            summary = run_ingestion(
                manifest_path=manifest_path,
                output_dir=output_dir,
                raw_dir=fixture_dir,
                verbose=False,
            )

            # Verify outputs
            assert summary["total_artifacts"] > 0
            assert summary["outcome_records"] > 0
            assert (output_dir / "artifacts.jsonl").exists()
            assert (output_dir / "outcomes_major_level.csv").exists()
            assert (output_dir / "provenance.jsonl").exists()
            assert (output_dir / "manifest_snapshot.yaml").exists()

            # Verify artifact format
            with open(output_dir / "artifacts.jsonl", encoding="utf-8") as f:
                first_line = f.readline()
                artifact = json.loads(first_line)
                assert "artifact_id" in artifact
                assert "text" in artifact
                assert artifact["type"] == "OUTCOME_MAJOR"

            # Verify CSV format
            with open(output_dir / "outcomes_major_level.csv", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) > 0
                assert "major" in rows[0]
                assert "employment_rate" in rows[0]
