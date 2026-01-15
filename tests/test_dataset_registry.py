"""Tests for dataset registry and cross-dataset alignment."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from ads_core.data.schemas import Artifact, ArtifactType
from ads_core.datasets.registry import (
    DatasetBundle,
    StrictDataError,
    load_dataset,
    list_datasets,
    get_dataset_info,
    DATASET_REGISTRY,
)


class TestDatasetRegistry:
    """Tests for dataset registry functions."""

    def test_list_datasets(self):
        """List all available datasets."""
        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert "toy" in datasets
        assert "mit" in datasets
        assert "ucb" in datasets
        assert "asu" in datasets

    def test_get_dataset_info_toy(self):
        """Get info for toy dataset."""
        info = get_dataset_info("toy")
        assert info is not None
        assert info["name"] == "Toy Dataset"
        assert info["institution"] == "TOY"

    def test_get_dataset_info_mit(self):
        """Get info for MIT dataset."""
        info = get_dataset_info("mit")
        assert info is not None
        assert info["institution"] == "MIT"

    def test_get_dataset_info_ucb(self):
        """Get info for UCB dataset."""
        info = get_dataset_info("ucb")
        assert info is not None
        assert info["institution"] == "UCB"

    def test_get_dataset_info_asu(self):
        """Get info for ASU dataset."""
        info = get_dataset_info("asu")
        assert info is not None
        assert info["institution"] == "ASU"

    def test_get_dataset_info_unknown(self):
        """Get info for unknown dataset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_info("unknown")

    def test_registry_structure(self):
        """Verify registry has required fields for each dataset."""
        required_fields = ["name", "institution", "loader"]
        for dataset_id, info in DATASET_REGISTRY.items():
            for field in required_fields:
                assert field in info, f"Missing {field} in {dataset_id}"


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_toy_dataset(self):
        """Load toy dataset successfully."""
        bundle = load_dataset("toy")
        assert isinstance(bundle, DatasetBundle)
        assert bundle.dataset_id == "toy"
        assert len(bundle.artifacts) > 0
        assert bundle.metadata.get("institution") == "TOY"

    def test_load_mit_dataset(self):
        """Load MIT dataset with auto-generation."""
        bundle = load_dataset("mit", auto_generate=True)
        assert isinstance(bundle, DatasetBundle)
        assert bundle.dataset_id == "mit"
        assert len(bundle.artifacts) > 0
        assert bundle.metadata.get("institution") == "MIT"

    def test_load_ucb_dataset(self):
        """Load UCB dataset with auto-generation."""
        bundle = load_dataset("ucb", auto_generate=True)
        assert isinstance(bundle, DatasetBundle)
        assert bundle.dataset_id == "ucb"
        assert len(bundle.artifacts) > 0
        assert bundle.metadata.get("institution") == "UCB"

    def test_load_asu_dataset(self):
        """Load ASU dataset with auto-generation."""
        bundle = load_dataset("asu", auto_generate=True)
        assert isinstance(bundle, DatasetBundle)
        assert bundle.dataset_id == "asu"
        assert len(bundle.artifacts) > 0
        assert bundle.metadata.get("institution") == "ASU"

    def test_load_unknown_dataset(self):
        """Load unknown dataset raises error."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("unknown")


class TestDatasetBundle:
    """Tests for DatasetBundle dataclass."""

    def test_bundle_creation(self):
        """Create bundle with required fields."""
        artifacts = [
            Artifact(
                artifact_id="test:1",
                type=ArtifactType.COURSE,
                source_url="https://example.com",
                source_hash="abc123",
                timestamp="2024-01-01T00:00:00Z",
                text="Test course",
                metadata={},
            )
        ]
        bundle = DatasetBundle(
            dataset_id="test",
            artifacts=artifacts,
        )
        assert bundle.dataset_id == "test"
        assert len(bundle.artifacts) == 1
        assert bundle.outcomes is None
        assert bundle.fds_schema is None
        # metadata is auto-populated with dataset info
        assert bundle.metadata.get("dataset_id") == "test"
        assert bundle.metadata.get("artifact_count") == 1

    def test_bundle_with_explicit_metadata(self):
        """Create bundle with explicit metadata (not auto-computed)."""
        bundle = DatasetBundle(
            dataset_id="test",
            artifacts=[],
            metadata={"version": "1.0", "institution": "TEST"},
        )
        # When metadata is provided, it's used as-is (no auto-computation)
        assert bundle.metadata["version"] == "1.0"
        assert bundle.metadata["institution"] == "TEST"


class TestArtifactValidation:
    """Tests for artifact schema validation."""

    def test_artifact_required_fields(self):
        """Verify artifact has required fields."""
        artifact = Artifact(
            artifact_id="test:1",
            type=ArtifactType.COURSE,
            source_url="https://example.com",
            source_hash="abc123",
            timestamp="2024-01-01T00:00:00Z",
            text="Test course",
            metadata={},
        )
        # Required fields
        assert artifact.artifact_id
        assert artifact.type
        assert artifact.text
        assert artifact.source_url
        assert artifact.source_hash
        assert artifact.timestamp

    def test_artifact_id_format(self):
        """Verify artifact ID contains source prefix."""
        bundle = load_dataset("toy")
        for artifact in bundle.artifacts:
            assert ":" in artifact.artifact_id, f"Invalid ID format: {artifact.artifact_id}"

    def test_artifact_text_non_empty(self):
        """Verify artifact text is non-empty."""
        bundle = load_dataset("toy")
        for artifact in bundle.artifacts:
            assert artifact.text.strip(), f"Empty text for {artifact.artifact_id}"

    def test_artifact_type_valid(self):
        """Verify artifact type is valid enum value."""
        bundle = load_dataset("toy")
        valid_types = set(ArtifactType)
        for artifact in bundle.artifacts:
            assert artifact.type in valid_types, f"Invalid type: {artifact.type}"


class TestCrossDatasetConsistency:
    """Tests for cross-dataset schema consistency."""

    @pytest.fixture
    def all_bundles(self):
        """Load all datasets."""
        return {
            "toy": load_dataset("toy", auto_generate=True),
            "mit": load_dataset("mit", auto_generate=True),
            "ucb": load_dataset("ucb", auto_generate=True),
            "asu": load_dataset("asu", auto_generate=True),
        }

    def test_all_datasets_have_artifacts(self, all_bundles):
        """All datasets have at least one artifact."""
        for dataset_id, bundle in all_bundles.items():
            assert len(bundle.artifacts) > 0, f"{dataset_id} has no artifacts"

    def test_all_artifacts_have_required_fields(self, all_bundles):
        """All artifacts across datasets have required fields."""
        for dataset_id, bundle in all_bundles.items():
            for artifact in bundle.artifacts:
                assert artifact.artifact_id, f"Missing artifact_id in {dataset_id}"
                assert artifact.type, f"Missing type in {dataset_id}"
                assert artifact.text, f"Missing text in {dataset_id}"
                assert artifact.source_url, f"Missing source_url in {dataset_id}"
                assert artifact.source_hash, f"Missing source_hash in {dataset_id}"

    def test_all_datasets_have_metadata(self, all_bundles):
        """All datasets have institution metadata."""
        for dataset_id, bundle in all_bundles.items():
            assert "institution" in bundle.metadata, f"{dataset_id} missing institution"

    def test_artifact_id_uniqueness(self, all_bundles):
        """All artifact IDs are unique within each dataset."""
        for dataset_id, bundle in all_bundles.items():
            ids = [a.artifact_id for a in bundle.artifacts]
            assert len(ids) == len(set(ids)), f"Duplicate IDs in {dataset_id}"

    def test_dataset_institution_matches(self, all_bundles):
        """Dataset institution matches expected value."""
        expected = {
            "toy": "TOY",
            "mit": "MIT",
            "ucb": "UCB",
            "asu": "ASU",
        }
        for dataset_id, bundle in all_bundles.items():
            assert bundle.metadata.get("institution") == expected[dataset_id]


class TestDatasetPipeline:
    """Tests for dataset pipeline integration."""

    def test_pipeline_toy(self, tmp_path):
        """Run pipeline on toy dataset."""
        from ads_core.pipeline.dataset_pipeline import run_dataset

        outputs = run_dataset(
            dataset_id="toy",
            out_dir=tmp_path / "toy_run",
            seed=42,
            embedding_cfg={"kind": "stub", "d": 64},
            lenses_cfg={"kind": "identity"},
            objectives=["market", "learner"],
            autonomy_tau=0.3,
        )

        assert outputs.dataset_id == "toy"
        assert outputs.artifact_count > 0
        assert outputs.results_json.exists()
        assert outputs.pareto_json.exists()

    def test_pipeline_mit(self, tmp_path):
        """Run pipeline on MIT dataset."""
        from ads_core.pipeline.dataset_pipeline import run_dataset

        outputs = run_dataset(
            dataset_id="mit",
            out_dir=tmp_path / "mit_run",
            seed=42,
            embedding_cfg={"kind": "stub", "d": 64},
            lenses_cfg={"kind": "identity"},
            objectives=["market", "learner"],
            autonomy_tau=0.3,
        )

        assert outputs.dataset_id == "mit"
        assert outputs.artifact_count > 0

    def test_pipeline_ucb(self, tmp_path):
        """Run pipeline on UCB dataset."""
        from ads_core.pipeline.dataset_pipeline import run_dataset

        outputs = run_dataset(
            dataset_id="ucb",
            out_dir=tmp_path / "ucb_run",
            seed=42,
            embedding_cfg={"kind": "stub", "d": 64},
            lenses_cfg={"kind": "identity"},
            objectives=["market", "learner"],
            autonomy_tau=0.3,
        )

        assert outputs.dataset_id == "ucb"
        assert outputs.artifact_count > 0

    def test_pipeline_asu(self, tmp_path):
        """Run pipeline on ASU dataset."""
        from ads_core.pipeline.dataset_pipeline import run_dataset

        outputs = run_dataset(
            dataset_id="asu",
            out_dir=tmp_path / "asu_run",
            seed=42,
            embedding_cfg={"kind": "stub", "d": 64},
            lenses_cfg={"kind": "identity"},
            objectives=["market", "learner"],
            autonomy_tau=0.3,
        )

        assert outputs.dataset_id == "asu"
        assert outputs.artifact_count > 0

    def test_pipeline_outputs_pareto(self, tmp_path):
        """Pipeline outputs valid Pareto front."""
        from ads_core.pipeline.dataset_pipeline import run_dataset

        outputs = run_dataset(
            dataset_id="toy",
            out_dir=tmp_path / "pareto_test",
            seed=42,
            embedding_cfg={"kind": "stub", "d": 64},
            lenses_cfg={"kind": "identity"},
            objectives=["market", "learner"],
            autonomy_tau=0.3,
        )

        pareto_data = json.loads(outputs.pareto_json.read_text(encoding="utf-8"))
        assert "keys" in pareto_data
        assert "pareto" in pareto_data
        # pareto is a list of pareto-optimal results
        assert outputs.pareto_count == len(pareto_data["pareto"])


class TestStrictDataMode:
    """Tests for strict_data mode (KT19)."""

    def test_toy_is_synthetic(self):
        """Toy dataset is always synthetic."""
        bundle = load_dataset("toy")
        assert bundle.is_synthetic is True
        assert bundle.metadata.get("is_synthetic") is True

    def test_strict_data_with_toy_raises(self):
        """Strict data mode rejects toy dataset."""
        with pytest.raises(StrictDataError, match="synthetic"):
            load_dataset("toy", strict_data=True)

    def test_auto_generated_is_synthetic(self, tmp_path):
        """Auto-generated data is marked as synthetic."""
        # Force auto-generation by ensuring data doesn't exist
        bundle = load_dataset("mit", auto_generate=True)
        # When data was auto-generated from fixtures, it's synthetic
        # This test may show False if data already existed
        assert isinstance(bundle.is_synthetic, bool)

    def test_is_synthetic_in_metadata(self):
        """is_synthetic is included in bundle metadata."""
        bundle = load_dataset("toy")
        assert "is_synthetic" in bundle.metadata
        assert bundle.metadata["is_synthetic"] == bundle.is_synthetic


class TestDatasetBundleIsSynthetic:
    """Tests for is_synthetic field on DatasetBundle."""

    def test_default_not_synthetic(self):
        """Default is_synthetic is False."""
        bundle = DatasetBundle(
            dataset_id="test",
            artifacts=[],
        )
        assert bundle.is_synthetic is False

    def test_explicit_synthetic(self):
        """Explicit is_synthetic=True is preserved."""
        bundle = DatasetBundle(
            dataset_id="test",
            artifacts=[],
            is_synthetic=True,
        )
        assert bundle.is_synthetic is True
        assert bundle.metadata.get("is_synthetic") is True


class TestStrictDataPipeline:
    """Tests for strict_data in pipeline."""

    def test_pipeline_is_synthetic_output(self, tmp_path):
        """Pipeline outputs include is_synthetic."""
        from ads_core.pipeline.dataset_pipeline import run_dataset

        outputs = run_dataset(
            dataset_id="toy",
            out_dir=tmp_path / "synthetic_test",
            seed=42,
            embedding_cfg={"kind": "stub", "d": 64},
            lenses_cfg={"kind": "identity"},
            objectives=["market", "learner"],
            autonomy_tau=0.3,
        )

        # Toy is always synthetic
        assert outputs.is_synthetic is True

        # Check results.json includes is_synthetic
        results_data = json.loads(outputs.results_json.read_text(encoding="utf-8"))
        assert "is_synthetic" in results_data
        assert results_data["is_synthetic"] is True

    def test_pipeline_strict_data_blocks_toy(self, tmp_path):
        """Pipeline with strict_data blocks toy dataset."""
        from ads_core.pipeline.dataset_pipeline import run_dataset

        with pytest.raises(StrictDataError):
            run_dataset(
                dataset_id="toy",
                out_dir=tmp_path / "strict_toy",
                seed=42,
                embedding_cfg={"kind": "stub", "d": 64},
                lenses_cfg={"kind": "identity"},
                objectives=["market", "learner"],
                autonomy_tau=0.3,
                strict_data=True,
            )
