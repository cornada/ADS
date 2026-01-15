"""Tests for MIT curriculum ingestion."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ads_core.data.schemas import ArtifactType
from ads_core.ingest.mit_catalog import (
    MitCatalogConnector,
    parse_catalog_html,
    create_catalog_fixture,
)
from ads_core.ingest.mit_ocw import (
    MitOcwConnector,
    create_mit_fixtures,
    extract_html_text,
)


class TestCatalogParser:
    """Test MIT catalog HTML parsing."""

    def test_parse_courseblock_html(self):
        """Test parsing standard courseblock HTML."""
        html = """
        <div class="courseblock">
        <p class="courseblocktitle"><strong>6.001 Introduction to Programming</strong></p>
        <p class="courseblockdesc">Learn to program with Python.</p>
        </div>
        """
        courses = parse_catalog_html(html)
        assert len(courses) == 1
        assert courses[0]["number"] == "6.001"
        assert "Introduction to Programming" in courses[0]["title"]
        assert "Learn to program" in courses[0]["description"]

    def test_parse_multiple_courses(self):
        """Test parsing multiple course blocks."""
        html = """
        <div class="courseblock">
        <p class="courseblocktitle"><strong>6.001 Course One</strong></p>
        <p class="courseblockdesc">Description one.</p>
        </div>
        <div class="courseblock">
        <p class="courseblocktitle"><strong>6.002 Course Two</strong></p>
        <p class="courseblockdesc">Description two.</p>
        </div>
        """
        courses = parse_catalog_html(html)
        assert len(courses) == 2
        assert courses[0]["number"] == "6.001"
        assert courses[1]["number"] == "6.002"

    def test_parse_empty_html(self):
        """Test parsing HTML with no courses."""
        courses = parse_catalog_html("<html><body>No courses</body></html>")
        assert courses == []


class TestMitCatalogConnector:
    """Test MIT catalog connector."""

    def test_connector_with_fixture(self):
        """Test connector using created fixture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir) / "fixtures"
            create_catalog_fixture(fixture_dir, dept_id="6")

            connector = MitCatalogConnector(
                snapshot_dir=fixture_dir,
                departments=["6"],
            )

            artifacts = list(connector.fetch())
            assert len(artifacts) > 0

            # Check artifact structure
            art = artifacts[0]
            assert art.artifact_id.startswith("mit_catalog:")
            assert art.type == ArtifactType.COURSE
            assert "course_number" in art.metadata
            assert "provenance" in art.metadata

    def test_connector_max_courses(self):
        """Test max_courses limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir) / "fixtures"
            create_catalog_fixture(fixture_dir, dept_id="6")

            connector = MitCatalogConnector(
                snapshot_dir=fixture_dir,
                departments=["6"],
                max_courses=3,
            )

            artifacts = list(connector.fetch())
            assert len(artifacts) <= 3

    def test_connector_source_name(self):
        """Test connector source name."""
        connector = MitCatalogConnector()
        assert connector.source_name == "mit_catalog"


class TestMitOcwConnector:
    """Test enhanced OCW connector."""

    def test_connector_with_fixtures(self):
        """Test OCW connector using fixtures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir) / "fixtures"
            create_mit_fixtures(fixture_dir)

            # Create minimal manifest
            manifest_path = Path(tmpdir) / "manifest.yaml"
            manifest = {
                "version": "1.0",
                "institution": "mit",
                "ocw_courses": [
                    {"number": "6.0001", "title": "Intro CS", "url": "https://ocw.mit.edu/..."},
                ],
                "missions": [
                    {"id": "mit_mission", "name": "MIT Mission", "url": "https://mit.edu/mission"},
                ],
            }
            manifest_path.write_text(
                __import__("yaml").dump(manifest),
                encoding="utf-8"
            )

            connector = MitOcwConnector(
                manifest_path=manifest_path,
                snapshot_dir=fixture_dir,
                include_missions=True,
            )

            artifacts = list(connector.fetch())
            assert len(artifacts) > 0

            # Check for mission artifacts
            mission_arts = [a for a in artifacts if a.type == ArtifactType.MISSION]
            assert len(mission_arts) >= 1

    def test_connector_source_name(self):
        """Test OCW connector source name."""
        connector = MitOcwConnector()
        assert connector.source_name == "mit_ocw_enhanced"


class TestFixtureCreation:
    """Test fixture creation functions."""

    def test_create_catalog_fixture(self):
        """Test catalog fixture creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir)
            create_catalog_fixture(fixture_dir, dept_id="6")

            assert (fixture_dir / "dept_6.html").exists()
            assert (fixture_dir / "meta.json").exists()

            # Check HTML content
            html = (fixture_dir / "dept_6.html").read_text(encoding="utf-8")
            assert "courseblock" in html
            assert "6.0001" in html

    def test_create_mit_fixtures(self):
        """Test complete MIT fixture creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir)
            create_mit_fixtures(fixture_dir)

            # Check OCW fixtures
            assert (fixture_dir / "ocw" / "courses.json").exists()

            # Check catalog fixtures
            assert (fixture_dir / "mit_catalog" / "dept_6.html").exists()

            # Check mission fixtures
            assert (fixture_dir / "mit_missions" / "mit_mission.txt").exists()
            assert (fixture_dir / "mit_missions" / "eecs_mission.txt").exists()

            # Check metadata
            assert (fixture_dir / "meta.json").exists()


class TestHtmlTextExtraction:
    """Test HTML text extraction."""

    def test_extract_text(self):
        """Test basic text extraction."""
        html = "<html><body><p>Hello world</p></body></html>"
        text = extract_html_text(html)
        assert "Hello world" in text

    def test_extract_text_strips_scripts(self):
        """Test that scripts are stripped."""
        html = "<html><body><script>bad()</script><p>Good text</p></body></html>"
        text = extract_html_text(html)
        assert "Good text" in text
        assert "bad" not in text


class TestIntegration:
    """Integration tests for MIT ingestion."""

    def test_full_ingestion_flow(self):
        """Test complete ingestion from fixtures to artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fixture_dir = tmpdir / "fixtures"
            output_dir = tmpdir / "output"

            # Create fixtures
            create_mit_fixtures(fixture_dir)

            # Create manifest
            manifest_path = tmpdir / "manifest.yaml"
            manifest = {
                "version": "1.0",
                "institution": "mit",
                "departments": {
                    "primary": [{"id": "6", "name": "EECS"}],
                },
                "ocw_courses": [
                    {"number": "6.0001", "title": "Intro CS"},
                    {"number": "6.006", "title": "Algorithms"},
                ],
                "missions": [
                    {"id": "mit_mission", "name": "MIT Mission", "url": ""},
                ],
                "settings": {"max_courses": None},
            }
            manifest_path.write_text(
                __import__("yaml").dump(manifest),
                encoding="utf-8"
            )

            # Run ingestion
            from ads_core.ingest.mit import run_ingestion

            summary = run_ingestion(
                manifest_path=manifest_path,
                output_dir=output_dir,
                snapshot_dir=fixture_dir,
                verbose=False,
            )

            # Verify outputs
            assert summary["total_artifacts"] > 0
            assert (output_dir / "artifacts.jsonl").exists()
            assert (output_dir / "provenance.jsonl").exists()
            assert (output_dir / "manifest_snapshot.yaml").exists()

            # Verify artifact format
            with open(output_dir / "artifacts.jsonl", encoding="utf-8") as f:
                first_line = f.readline()
                artifact = json.loads(first_line)
                assert "artifact_id" in artifact
                assert "text" in artifact
                assert "type" in artifact

    def test_ingestion_with_max_courses(self):
        """Test ingestion with course limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fixture_dir = tmpdir / "fixtures"
            output_dir = tmpdir / "output"

            create_mit_fixtures(fixture_dir)

            manifest_path = tmpdir / "manifest.yaml"
            manifest = {
                "version": "1.0",
                "institution": "mit",
                "departments": {"primary": [{"id": "6"}]},
                "ocw_courses": [],
                "missions": [],
            }
            manifest_path.write_text(
                __import__("yaml").dump(manifest),
                encoding="utf-8"
            )

            from ads_core.ingest.mit import run_ingestion

            summary = run_ingestion(
                manifest_path=manifest_path,
                output_dir=output_dir,
                snapshot_dir=fixture_dir,
                max_courses=2,
                verbose=False,
            )

            # Should be limited
            assert summary["catalog_courses"] <= 2
