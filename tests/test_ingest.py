"""Tests for data ingestion connectors and provenance tracking."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ads_core.data.schemas import Artifact, ArtifactType
from ads_core.ingest.base import (
    BaseConnector,
    ProvenanceInfo,
    create_provenance,
    normalize_text,
)
from ads_core.ingest.onet import OnetConnector, create_onet_fixture
from ads_core.ingest.ocw import OcwConnector, create_ocw_fixture
from ads_core.ingest.toy_dataset import build_toy_artifacts
from ads_core.utils.hashing import sha256_text


class TestProvenanceTracking:
    """Tests for provenance utilities."""

    def test_create_provenance_basic(self):
        """Test basic provenance creation."""
        prov = create_provenance(
            source_url="https://example.com/data",
            raw_text="Hello world",
        )
        assert prov.source_url == "https://example.com/data"
        assert prov.raw_text_hash == sha256_text("Hello world")
        assert isinstance(prov.fetched_at, datetime)
        assert prov.fetched_at.tzinfo is not None  # timezone-aware

    def test_create_provenance_with_extras(self):
        """Test provenance with extra metadata."""
        prov = create_provenance(
            source_url="https://example.com",
            raw_text="test",
            version="1.0",
            custom_field="value",
        )
        d = prov.to_dict()
        assert d["version"] == "1.0"
        assert d["custom_field"] == "value"

    def test_provenance_to_dict(self):
        """Test provenance serialization."""
        prov = create_provenance(
            source_url="https://test.com",
            raw_text="sample",
            snapshot_path=Path("/data/snap.json"),
        )
        d = prov.to_dict()
        assert "source_url" in d
        assert "fetched_at" in d
        assert "raw_text_hash" in d
        # Path string format varies by OS
        assert "snap.json" in d["snapshot_path"]


class TestNormalization:
    """Tests for text normalization."""

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "  Hello   world  \n\t test  "
        assert normalize_text(text) == "Hello world test"

    def test_normalize_empty(self):
        """Test empty string normalization."""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""


class TestOnetConnector:
    """Tests for O*NET connector."""

    @pytest.fixture
    def onet_fixture_dir(self, tmp_path):
        """Create a temporary O*NET fixture."""
        create_onet_fixture(tmp_path)
        return tmp_path

    def test_onet_loads_from_fixture(self, onet_fixture_dir):
        """Test O*NET connector loads fixture data."""
        connector = OnetConnector(snapshot_path=onet_fixture_dir)
        artifacts = connector.fetch_all()

        assert len(artifacts) > 0
        assert all(isinstance(a, Artifact) for a in artifacts)
        assert all(a.type == ArtifactType.JOB_ROLE for a in artifacts)

    def test_onet_artifact_structure(self, onet_fixture_dir):
        """Test O*NET artifacts have required fields."""
        connector = OnetConnector(snapshot_path=onet_fixture_dir, max_occupations=1)
        artifacts = connector.fetch_all()

        assert len(artifacts) == 1
        art = artifacts[0]

        # Check required fields
        assert art.artifact_id.startswith("onet:")
        assert art.source_url is not None
        assert art.source_hash is not None
        assert art.timestamp is not None
        assert len(art.text) > 0

        # Check metadata
        assert "onet_soc_code" in art.metadata
        assert "title" in art.metadata
        assert "provenance" in art.metadata

    def test_onet_provenance_in_metadata(self, onet_fixture_dir):
        """Test O*NET artifacts include provenance info."""
        connector = OnetConnector(snapshot_path=onet_fixture_dir, max_occupations=1)
        art = connector.fetch_all()[0]

        prov = art.metadata["provenance"]
        assert "source_url" in prov
        assert "fetched_at" in prov
        assert "raw_text_hash" in prov
        assert "onet_version" in prov

    def test_onet_max_occupations(self, onet_fixture_dir):
        """Test max_occupations limit."""
        connector = OnetConnector(snapshot_path=onet_fixture_dir, max_occupations=3)
        artifacts = connector.fetch_all()
        assert len(artifacts) == 3

    def test_onet_source_name(self):
        """Test O*NET source name includes version."""
        connector = OnetConnector()
        assert "onet" in connector.source_name
        assert "_" in connector.source_name  # version separator


class TestOcwConnector:
    """Tests for MIT OCW connector."""

    @pytest.fixture
    def ocw_fixture_dir(self, tmp_path):
        """Create a temporary OCW fixture."""
        create_ocw_fixture(tmp_path)
        return tmp_path

    def test_ocw_loads_from_fixture(self, ocw_fixture_dir):
        """Test OCW connector loads fixture data."""
        connector = OcwConnector(snapshot_path=ocw_fixture_dir)
        artifacts = connector.fetch_all()

        assert len(artifacts) > 0
        assert all(isinstance(a, Artifact) for a in artifacts)
        assert all(a.type == ArtifactType.COURSE for a in artifacts)

    def test_ocw_artifact_structure(self, ocw_fixture_dir):
        """Test OCW artifacts have required fields."""
        connector = OcwConnector(snapshot_path=ocw_fixture_dir, max_courses=1)
        artifacts = connector.fetch_all()

        assert len(artifacts) == 1
        art = artifacts[0]

        # Check required fields
        assert art.artifact_id.startswith("ocw:")
        assert art.source_url is not None
        assert art.source_hash is not None
        assert art.timestamp is not None
        assert len(art.text) > 0

        # Check metadata
        assert "course_number" in art.metadata
        assert "title" in art.metadata
        assert "provenance" in art.metadata

    def test_ocw_department_filter(self, ocw_fixture_dir):
        """Test OCW department filtering."""
        # Get all courses first
        all_connector = OcwConnector(snapshot_path=ocw_fixture_dir)
        all_artifacts = all_connector.fetch_all()

        # Filter to department 6 (EECS)
        filtered_connector = OcwConnector(
            snapshot_path=ocw_fixture_dir, departments=["6"]
        )
        filtered_artifacts = filtered_connector.fetch_all()

        # Should have fewer or equal courses
        assert len(filtered_artifacts) <= len(all_artifacts)
        # All filtered should be from dept 6
        for art in filtered_artifacts:
            assert art.metadata.get("department") == "6"

    def test_ocw_max_courses(self, ocw_fixture_dir):
        """Test max_courses limit."""
        connector = OcwConnector(snapshot_path=ocw_fixture_dir, max_courses=2)
        artifacts = connector.fetch_all()
        assert len(artifacts) == 2


class TestToyDataset:
    """Tests for built-in toy dataset."""

    def test_toy_builds_artifacts(self):
        """Test toy dataset builds valid artifacts."""
        artifacts = build_toy_artifacts()

        assert len(artifacts) > 0
        assert all(isinstance(a, Artifact) for a in artifacts)

    def test_toy_has_all_types(self):
        """Test toy dataset includes various artifact types."""
        artifacts = build_toy_artifacts()
        types = {a.type for a in artifacts}

        assert ArtifactType.MISSION in types
        assert ArtifactType.COURSE in types
        assert ArtifactType.JOB_ROLE in types
        assert ArtifactType.LEARNER_PROFILE in types

    def test_toy_provenance_fields(self):
        """Test toy artifacts have provenance fields."""
        artifacts = build_toy_artifacts()

        for art in artifacts:
            assert art.source_url is not None
            assert art.source_url.startswith("toy://")
            assert art.source_hash is not None
            assert art.timestamp is not None


class TestCreateFixtures:
    """Tests for fixture creation utilities."""

    def test_create_onet_fixture(self, tmp_path):
        """Test O*NET fixture creation."""
        create_onet_fixture(tmp_path)

        # Check files created
        assert (tmp_path / "occupations.json").exists()
        assert (tmp_path / "skills.json").exists()
        assert (tmp_path / "tasks.json").exists()
        assert (tmp_path / "meta.json").exists()

        # Check content is valid JSON
        occs = json.loads((tmp_path / "occupations.json").read_text())
        assert isinstance(occs, list)
        assert len(occs) > 0

    def test_create_ocw_fixture(self, tmp_path):
        """Test OCW fixture creation."""
        create_ocw_fixture(tmp_path)

        # Check files created
        assert (tmp_path / "courses.json").exists()
        assert (tmp_path / "meta.json").exists()

        # Check content is valid JSON
        courses = json.loads((tmp_path / "courses.json").read_text())
        assert isinstance(courses, list)
        assert len(courses) > 0


class TestArtifactIntegrity:
    """Tests for artifact data integrity."""

    def test_artifact_id_uniqueness(self, tmp_path):
        """Test that artifact IDs are unique within a source."""
        create_onet_fixture(tmp_path / "onet")
        create_ocw_fixture(tmp_path / "ocw")

        onet = OnetConnector(snapshot_path=tmp_path / "onet")
        ocw = OcwConnector(snapshot_path=tmp_path / "ocw")

        onet_ids = [a.artifact_id for a in onet.fetch_all()]
        ocw_ids = [a.artifact_id for a in ocw.fetch_all()]

        # Check uniqueness within each source
        assert len(onet_ids) == len(set(onet_ids)), "O*NET IDs not unique"
        assert len(ocw_ids) == len(set(ocw_ids)), "OCW IDs not unique"

        # Check no overlap between sources (due to prefix)
        assert not set(onet_ids) & set(ocw_ids), "ID collision between sources"

    def test_source_hash_reproducibility(self, tmp_path):
        """Test that source_hash is deterministic."""
        create_onet_fixture(tmp_path)

        connector1 = OnetConnector(snapshot_path=tmp_path, max_occupations=1)
        connector2 = OnetConnector(snapshot_path=tmp_path, max_occupations=1)

        art1 = connector1.fetch_all()[0]
        art2 = connector2.fetch_all()[0]

        # Same data should produce same hash
        assert art1.source_hash == art2.source_hash
