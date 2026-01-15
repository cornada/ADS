"""Tests for ASU FDS and outcomes ingestion."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from ads_core.data.schemas import ArtifactType
from ads_core.ingest.asu_fds import (
    AsuFdsConnector,
    SurveyQuestion,
    SurveySection,
    FDSSchema,
    create_fds_fixture,
)
from ads_core.ingest.asu_outcomes import (
    AsuOutcomesConnector,
    OutcomeSummary,
    parse_outcomes_csv,
    export_outcomes_csv,
    create_outcomes_fixture,
)


class TestSurveyQuestion:
    """Test SurveyQuestion dataclass."""

    def test_to_dict(self):
        """Test conversion to dict."""
        q = SurveyQuestion(
            question_id="Q1",
            section="employment_status",
            text="What is your employment status?",
            question_type="single_choice",
            options=["Employed", "Seeking"],
            required=True,
        )
        d = q.to_dict()
        assert d["question_id"] == "Q1"
        assert d["question_type"] == "single_choice"
        assert len(d["options"]) == 2

    def test_to_dict_with_depends_on(self):
        """Test dict with branching logic."""
        q = SurveyQuestion(
            question_id="Q2",
            section="employment_details",
            text="What is your job title?",
            question_type="text",
            depends_on="Q1=Employed",
        )
        d = q.to_dict()
        assert d["depends_on"] == "Q1=Employed"


class TestSurveySection:
    """Test SurveySection dataclass."""

    def test_to_dict(self):
        """Test conversion to dict."""
        section = SurveySection(
            section_id="employment_status",
            name="Employment Status",
            description="Determine current activity",
        )
        section.questions.append(
            SurveyQuestion(
                question_id="Q1",
                section="employment_status",
                text="Status?",
                question_type="single_choice",
            )
        )
        d = section.to_dict()
        assert d["section_id"] == "employment_status"
        assert len(d["questions"]) == 1


class TestFDSSchema:
    """Test FDSSchema class."""

    def test_to_text_summary(self):
        """Test text summary generation."""
        schema = FDSSchema(
            version="1.0",
            source_url="https://example.com/fds",
        )
        schema.sections.append(
            SurveySection(
                section_id="employment",
                name="Employment",
                description="Employment questions",
            )
        )
        text = schema.to_text_summary()
        assert "First Destination Survey" in text
        assert "Employment" in text


class TestAsuFdsConnector:
    """Test ASU FDS connector."""

    def test_connector_with_fixture(self):
        """Test connector using fixture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir)
            create_fds_fixture(fixture_dir)

            connector = AsuFdsConnector(raw_dir=fixture_dir)
            schema = connector.load_schema()

            assert schema.version == "1.0"
            assert len(schema.sections) > 0

    def test_fetch_artifacts(self):
        """Test fetching artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir)
            create_fds_fixture(fixture_dir)

            connector = AsuFdsConnector(raw_dir=fixture_dir)
            artifacts = list(connector.fetch())

            assert len(artifacts) > 0
            # Should have schema artifact + section artifacts
            schema_arts = [a for a in artifacts if a.artifact_id == "asu_fds:schema"]
            assert len(schema_arts) == 1
            assert schema_arts[0].type == ArtifactType.SURVEY_SCHEMA

    def test_connector_source_name(self):
        """Test connector source name."""
        connector = AsuFdsConnector()
        assert connector.source_name == "asu_fds"


class TestOutcomeSummary:
    """Test OutcomeSummary dataclass."""

    def test_to_dict(self):
        """Test conversion to dict."""
        summary = OutcomeSummary(
            college="FSE",
            college_name="Fulton Schools of Engineering",
            academic_year="2015-2016",
            employment_rate=0.72,
            career_outcomes_rate=0.90,
        )
        d = summary.to_dict()
        assert d["college"] == "FSE"
        assert d["employment_rate"] == 0.72
        assert "grad_school_rate" not in d  # None values excluded

    def test_to_text(self):
        """Test text generation."""
        summary = OutcomeSummary(
            college="FSE",
            college_name="Fulton Schools of Engineering",
            academic_year="2015-2016",
            employment_rate=0.72,
            grad_school_rate=0.18,
            career_outcomes_rate=0.90,
            median_salary=62000,
            n_respondents=1850,
        )
        text = summary.to_text()
        assert "Fulton Schools of Engineering" in text
        assert "2015-2016" in text
        assert "72.0%" in text
        assert "$62,000" in text


class TestOutcomesParsing:
    """Test outcomes CSV parsing."""

    def test_parse_basic_csv(self):
        """Test parsing basic CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text(
                "college,college_name,academic_year,employment_rate,career_outcomes_rate\n"
                "FSE,Engineering,2015-2016,0.72,0.90\n"
                "WPC,Business,2015-2016,0.78,0.86\n",
                encoding="utf-8"
            )

            records = parse_outcomes_csv(csv_path)
            assert len(records) == 2
            assert records[0].college == "FSE"
            assert records[0].employment_rate == 0.72

    def test_parse_percentage_format(self):
        """Test parsing percentages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text(
                "college,college_name,academic_year,employment_rate\n"
                "FSE,Engineering,2015-2016,72%\n",
                encoding="utf-8"
            )

            records = parse_outcomes_csv(csv_path)
            assert len(records) == 1
            assert records[0].employment_rate == 0.72


class TestAsuOutcomesConnector:
    """Test ASU outcomes connector."""

    def test_connector_with_fixture(self):
        """Test connector using fixture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir)
            create_outcomes_fixture(fixture_dir)

            connector = AsuOutcomesConnector(raw_dir=fixture_dir)
            records = connector.load_records()

            assert len(records) > 0

    def test_fetch_artifacts(self):
        """Test fetching artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir)
            create_outcomes_fixture(fixture_dir)

            connector = AsuOutcomesConnector(raw_dir=fixture_dir)
            artifacts = list(connector.fetch())

            assert len(artifacts) > 0
            assert all(a.type == ArtifactType.OUTCOME_SUMMARY for a in artifacts)

    def test_max_records(self):
        """Test max_records limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir)
            create_outcomes_fixture(fixture_dir)

            connector = AsuOutcomesConnector(
                raw_dir=fixture_dir,
                max_records=3,
            )
            artifacts = list(connector.fetch())

            assert len(artifacts) == 3

    def test_connector_source_name(self):
        """Test connector source name."""
        connector = AsuOutcomesConnector()
        assert connector.source_name == "asu_outcomes"


class TestFixtureCreation:
    """Test fixture creation."""

    def test_create_fds_fixture(self):
        """Test FDS fixture creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir)
            create_fds_fixture(fixture_dir)

            assert (fixture_dir / "fds_schema.json").exists()
            assert (fixture_dir / "fds_meta.json").exists()

            # Check schema content
            schema = json.loads(
                (fixture_dir / "fds_schema.json").read_text(encoding="utf-8")
            )
            assert "sections" in schema
            assert len(schema["sections"]) > 0

            # Check that questions exist
            total_questions = sum(
                len(s.get("questions", [])) for s in schema["sections"]
            )
            assert total_questions > 0

    def test_create_outcomes_fixture(self):
        """Test outcomes fixture creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir)
            create_outcomes_fixture(fixture_dir)

            assert (fixture_dir / "outcomes_summary.csv").exists()
            assert (fixture_dir / "outcomes_meta.json").exists()

            # Check CSV content
            with open(fixture_dir / "outcomes_summary.csv", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) > 0
            assert "college" in rows[0]
            assert "employment_rate" in rows[0]


class TestExportCsv:
    """Test CSV export."""

    def test_export_outcomes_csv(self):
        """Test exporting outcomes to CSV."""
        records = [
            OutcomeSummary(
                college="FSE",
                college_name="Engineering",
                academic_year="2015-2016",
                employment_rate=0.72,
                career_outcomes_rate=0.90,
            ),
            OutcomeSummary(
                college="WPC",
                college_name="Business",
                academic_year="2015-2016",
                employment_rate=0.78,
                career_outcomes_rate=0.86,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "outcomes.csv"
            count = export_outcomes_csv(records, output_path)

            assert count == 2
            assert output_path.exists()

            with open(output_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["college"] == "FSE"


class TestIntegration:
    """Integration tests for ASU ingestion."""

    def test_full_ingestion_flow(self):
        """Test complete ingestion flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fixture_dir = tmpdir / "fixtures"
            output_dir = tmpdir / "output"

            # Create fixtures
            create_fds_fixture(fixture_dir)
            create_outcomes_fixture(fixture_dir)

            # Create manifest
            manifest_path = tmpdir / "manifest.yaml"
            manifest = {
                "version": "1.0",
                "institution": "asu",
                "sources": {
                    "fds_survey_preview": {"url": "https://uoeee.asu.edu/fds"},
                    "career_outcomes_1516": {"url": "https://provost.asu.edu/outcomes"},
                },
                "fds_sections": [],
            }
            manifest_path.write_text(
                __import__("yaml").dump(manifest),
                encoding="utf-8"
            )

            # Run ingestion
            from ads_core.ingest.asu import run_ingestion

            summary = run_ingestion(
                manifest_path=manifest_path,
                output_dir=output_dir,
                raw_dir=fixture_dir,
                verbose=False,
            )

            # Verify outputs
            assert summary["total_artifacts"] > 0
            assert (output_dir / "fds_schema.json").exists()
            assert (output_dir / "outcomes_summary.csv").exists()
            assert (output_dir / "artifacts.jsonl").exists()
            assert (output_dir / "provenance.jsonl").exists()

            # Verify FDS schema
            schema = json.loads(
                (output_dir / "fds_schema.json").read_text(encoding="utf-8")
            )
            assert "sections" in schema

            # Verify artifacts
            with open(output_dir / "artifacts.jsonl", encoding="utf-8") as f:
                first_line = f.readline()
                artifact = json.loads(first_line)
                assert "artifact_id" in artifact
                assert "text" in artifact
