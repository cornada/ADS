"""Dataset registry for ADS.

Provides unified access to institutional datasets with:
- Dataset discovery and loading
- Schema validation
- Metadata standardization
- Fixture fallback for reproducibility
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ads_core.data.schemas import Artifact, ArtifactType


# Dataset registry configuration
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "toy": {
        "name": "Toy Dataset",
        "institution": "TOY",
        "description": "Built-in toy dataset for testing",
        "loader": "toy",
        "artifacts_path": None,  # Generated dynamically
    },
    "mit": {
        "name": "MIT Curriculum",
        "institution": "MIT",
        "description": "MIT course catalog and OCW courses",
        "loader": "jsonl",
        "artifacts_path": "data/processed/mit/artifacts.jsonl",
        "fixture_path": "data/fixtures/mit",
        "manifest_path": "data/manifests/mit_sources.yaml",
    },
    "ucb": {
        "name": "UC Berkeley Outcomes",
        "institution": "UCB",
        "description": "UC Berkeley First Destination Survey outcomes",
        "loader": "jsonl",
        "artifacts_path": "data/processed/ucb/artifacts.jsonl",
        "outcomes_path": "data/processed/ucb/outcomes_major_level.csv",
        "fixture_path": "data/fixtures/ucb",
        "manifest_path": "data/manifests/ucb_sources.yaml",
    },
    "asu": {
        "name": "ASU FDS + Outcomes",
        "institution": "ASU",
        "description": "ASU First Destination Survey schema and career outcomes",
        "loader": "jsonl",
        "artifacts_path": "data/processed/asu/artifacts.jsonl",
        "outcomes_path": "data/processed/asu/outcomes_summary.csv",
        "fds_schema_path": "data/processed/asu/fds_schema.json",
        "fixture_path": "data/fixtures/asu",
        "manifest_path": "data/manifests/asu_sources.yaml",
    },
}


class StrictDataError(Exception):
    """Raised when strict_data mode is enabled and processed data is missing.

    This error indicates that a paper experiment is being run without properly
    ingested data, which could lead to invalid results based on synthetic fixtures.
    """

    pass


@dataclass
class DatasetBundle:
    """Container for a loaded dataset with artifacts and metadata.

    Attributes:
        dataset_id: Identifier of the dataset (toy, mit, ucb, asu)
        artifacts: List of loaded artifacts
        outcomes: Optional DataFrame with career outcomes
        fds_schema: Optional FDS survey schema
        metadata: Additional metadata dict
        is_synthetic: True if dataset is toy/fixture-based (not real ingested data)
    """

    dataset_id: str
    artifacts: List[Artifact]
    outcomes: Optional[pd.DataFrame] = None
    fds_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_synthetic: bool = False

    def __post_init__(self):
        """Populate metadata after initialization."""
        if not self.metadata:
            self.metadata = self._compute_metadata()
        # Always include is_synthetic in metadata for reports
        self.metadata["is_synthetic"] = self.is_synthetic

    def _compute_metadata(self) -> Dict[str, Any]:
        """Compute dataset metadata from artifacts."""
        artifact_types = {}
        for art in self.artifacts:
            t = art.type.value if isinstance(art.type, ArtifactType) else str(art.type)
            artifact_types[t] = artifact_types.get(t, 0) + 1

        institutions = set()
        for art in self.artifacts:
            inst = art.metadata.get("institution", "UNKNOWN")
            institutions.add(inst)

        return {
            "dataset_id": self.dataset_id,
            "artifact_count": len(self.artifacts),
            "artifact_types": artifact_types,
            "institutions": sorted(institutions),
            "has_outcomes": self.outcomes is not None,
            "outcomes_count": len(self.outcomes) if self.outcomes is not None else 0,
            "has_fds_schema": self.fds_schema is not None,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_artifacts_by_type(self, artifact_type: ArtifactType) -> List[Artifact]:
        """Filter artifacts by type."""
        return [a for a in self.artifacts if a.type == artifact_type]

    def get_course_artifacts(self) -> List[Artifact]:
        """Get all course-type artifacts."""
        return self.get_artifacts_by_type(ArtifactType.COURSE)

    def get_outcome_artifacts(self) -> List[Artifact]:
        """Get all outcome-type artifacts."""
        outcome_types = {ArtifactType.OUTCOME_MAJOR, ArtifactType.OUTCOME_SUMMARY}
        return [a for a in self.artifacts if a.type in outcome_types]


def list_datasets() -> List[str]:
    """List all registered dataset IDs."""
    return list(DATASET_REGISTRY.keys())


def get_dataset_info(dataset_id: str) -> Dict[str, Any]:
    """Get registry info for a dataset."""
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_id}. Available: {list_datasets()}")
    return DATASET_REGISTRY[dataset_id].copy()


def validate_artifact(artifact: Artifact) -> List[str]:
    """Validate an artifact against the unified schema.

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Required fields
    if not artifact.artifact_id:
        errors.append("Missing artifact_id")
    if not artifact.text or not artifact.text.strip():
        errors.append("Empty or missing text")

    # Provenance fields
    if not artifact.source_url:
        errors.append("Missing source_url")

    # Metadata requirements
    meta = artifact.metadata or {}
    if "provenance" not in meta:
        errors.append("Missing provenance in metadata")

    return errors


def _load_toy_dataset() -> DatasetBundle:
    """Load the built-in toy dataset.

    The toy dataset is always synthetic (generated programmatically).
    """
    from ads_core.ingest.toy_dataset import build_toy_artifacts

    artifacts = build_toy_artifacts()

    # Add institution metadata
    for art in artifacts:
        if "institution" not in art.metadata:
            art.metadata["institution"] = "TOY"

    return DatasetBundle(
        dataset_id="toy",
        artifacts=artifacts,
        metadata={
            "institution": "TOY",
            "description": "Built-in toy dataset for testing",
        },
        is_synthetic=True,  # Toy dataset is always synthetic
    )


def _load_jsonl_artifacts(path: Path) -> List[Artifact]:
    """Load artifacts from JSONL file."""
    artifacts = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            # Convert type string to enum
            type_str = data.get("type", "COURSE")
            try:
                artifact_type = ArtifactType(type_str)
            except ValueError:
                artifact_type = ArtifactType.COURSE

            # Parse timestamp if present
            timestamp = None
            if data.get("timestamp"):
                try:
                    timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            artifact = Artifact(
                artifact_id=data.get("artifact_id", ""),
                type=artifact_type,
                source_url=data.get("source_url"),
                source_hash=data.get("source_hash"),
                timestamp=timestamp,
                text=data.get("text", ""),
                metadata=data.get("metadata", {}),
            )
            artifacts.append(artifact)

    return artifacts


def _load_outcomes_csv(path: Path) -> pd.DataFrame:
    """Load outcomes CSV into DataFrame."""
    return pd.read_csv(path)


def _load_fds_schema(path: Path) -> Dict[str, Any]:
    """Load FDS schema JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dataset_processed(dataset_id: str, base_dir: Path) -> tuple[bool, bool]:
    """Ensure dataset is processed, running ingestion if needed.

    Returns:
        Tuple of (success, was_auto_generated) where was_auto_generated is True
        if the dataset was created from fixtures during this call.
    """
    info = DATASET_REGISTRY[dataset_id]
    artifacts_path = info.get("artifacts_path")

    if not artifacts_path:
        return True, False  # No processing needed (e.g., toy)

    full_path = base_dir / artifacts_path
    if full_path.exists():
        return True, False  # Data already exists (not auto-generated now)

    # Try to run ingestion from fixtures
    print(f"Dataset {dataset_id} not found at {full_path}, running ingestion from fixtures...")

    try:
        if dataset_id == "mit":
            from ads_core.ingest.mit import run_ingestion, ensure_fixtures
            fixture_dir = base_dir / "data" / "fixtures"
            ensure_fixtures(fixture_dir)
            manifest_path = base_dir / info["manifest_path"]
            output_dir = base_dir / "data" / "processed" / "mit"
            run_ingestion(manifest_path, output_dir, fixture_dir, verbose=False)
        elif dataset_id == "ucb":
            from ads_core.ingest.ucb import run_ingestion
            from ads_core.ingest.ucb_outcomes import create_ucb_fixture
            fixture_dir = base_dir / "data" / "fixtures" / "ucb"
            if not fixture_dir.exists():
                create_ucb_fixture(fixture_dir)
            manifest_path = base_dir / info["manifest_path"]
            output_dir = base_dir / "data" / "processed" / "ucb"
            run_ingestion(manifest_path, output_dir, fixture_dir, verbose=False)
        elif dataset_id == "asu":
            from ads_core.ingest.asu import run_ingestion, ensure_fixtures
            fixture_dir = base_dir / "data" / "fixtures" / "asu"
            ensure_fixtures(fixture_dir)
            manifest_path = base_dir / info["manifest_path"]
            output_dir = base_dir / "data" / "processed" / "asu"
            run_ingestion(manifest_path, output_dir, verbose=False)
        return True, True  # Success, was auto-generated from fixtures
    except Exception as e:
        print(f"Warning: Could not auto-generate dataset {dataset_id}: {e}")
        return False, False


def load_dataset(
    dataset_id: str,
    data_dir: Optional[Path] = None,
    auto_generate: bool = True,
    strict_data: bool = False,
) -> DatasetBundle:
    """Load a dataset by ID.

    Args:
        dataset_id: Dataset identifier (toy, mit, ucb, asu)
        data_dir: Base data directory (defaults to project root)
        auto_generate: If True, run ingestion from fixtures if artifacts not found
        strict_data: If True, fail when processed data is missing instead of
            auto-generating from fixtures. Use this for paper experiments to
            ensure you're using real ingested data, not synthetic fixtures.

    Returns:
        DatasetBundle with artifacts and optional outcomes. The `is_synthetic`
        field indicates whether the data is from fixtures (True) or real
        ingestion (False).

    Raises:
        ValueError: If dataset_id is unknown
        FileNotFoundError: If dataset files not found and auto_generate fails
        StrictDataError: If strict_data=True and processed data is missing
    """
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_id}. Available: {list_datasets()}")

    info = DATASET_REGISTRY[dataset_id]

    # Determine base directory
    if data_dir is None:
        # Default to project root (4 levels up from this file)
        data_dir = Path(__file__).parent.parent.parent.parent

    # Special case for toy dataset
    if info["loader"] == "toy":
        if strict_data:
            raise StrictDataError(
                f"Dataset '{dataset_id}' is synthetic (toy dataset). "
                "Cannot use strict_data mode with toy dataset. "
                "Use a real dataset (mit, ucb, asu) with proper ingestion for paper experiments."
            )
        return _load_toy_dataset()

    # Check if processed data exists before potentially auto-generating
    artifacts_path = data_dir / info["artifacts_path"]
    data_existed_before = artifacts_path.exists()

    # In strict mode, fail immediately if data doesn't exist
    if strict_data and not data_existed_before:
        raise StrictDataError(
            f"Dataset '{dataset_id}' processed data not found at {artifacts_path}. "
            f"strict_data=True requires pre-ingested data. "
            f"Run the ingestion first: python -m ads_core.ingest.{dataset_id} "
            f"or set strict_data=False to use fixtures (not recommended for papers)."
        )

    # Track if data was auto-generated from fixtures
    is_synthetic = False
    if auto_generate and not data_existed_before:
        success, was_auto_generated = _ensure_dataset_processed(dataset_id, data_dir)
        is_synthetic = was_auto_generated

    # Load artifacts
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"Dataset artifacts not found: {artifacts_path}. "
            f"Run the ingestion first: python -m ads_core.ingest.{dataset_id}"
        )

    artifacts = _load_jsonl_artifacts(artifacts_path)

    # Add institution to metadata if missing
    institution = info["institution"]
    for art in artifacts:
        if "institution" not in art.metadata:
            art.metadata["institution"] = institution

    # Load outcomes if available
    outcomes = None
    outcomes_path = info.get("outcomes_path")
    if outcomes_path:
        full_outcomes_path = data_dir / outcomes_path
        if full_outcomes_path.exists():
            outcomes = _load_outcomes_csv(full_outcomes_path)

    # Load FDS schema if available
    fds_schema = None
    fds_path = info.get("fds_schema_path")
    if fds_path:
        full_fds_path = data_dir / fds_path
        if full_fds_path.exists():
            fds_schema = _load_fds_schema(full_fds_path)

    return DatasetBundle(
        dataset_id=dataset_id,
        artifacts=artifacts,
        outcomes=outcomes,
        fds_schema=fds_schema,
        metadata={
            "institution": institution,
            "name": info["name"],
            "description": info["description"],
            "artifacts_path": str(artifacts_path),
        },
        is_synthetic=is_synthetic,
    )


def get_dataset_stats(dataset_id: str, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Get statistics for a dataset without fully loading it.

    Args:
        dataset_id: Dataset identifier
        data_dir: Base data directory

    Returns:
        Dict with dataset statistics
    """
    bundle = load_dataset(dataset_id, data_dir)

    # Count by type
    type_counts = {}
    for art in bundle.artifacts:
        t = art.type.value if isinstance(art.type, ArtifactType) else str(art.type)
        type_counts[t] = type_counts.get(t, 0) + 1

    # Get unique institutions
    institutions = set()
    for art in bundle.artifacts:
        inst = art.metadata.get("institution", "UNKNOWN")
        institutions.add(inst)

    stats = {
        "dataset_id": dataset_id,
        "artifact_count": len(bundle.artifacts),
        "type_distribution": type_counts,
        "institutions": sorted(institutions),
        "has_outcomes": bundle.outcomes is not None,
        "outcomes_count": len(bundle.outcomes) if bundle.outcomes is not None else 0,
        "has_fds_schema": bundle.fds_schema is not None,
    }

    # Add outcome stats if available
    if bundle.outcomes is not None and len(bundle.outcomes) > 0:
        df = bundle.outcomes
        if "employment_rate" in df.columns:
            stats["mean_employment_rate"] = float(df["employment_rate"].mean())
        if "median_salary" in df.columns:
            stats["mean_median_salary"] = float(df["median_salary"].mean())

    return stats
