"""MIT Curriculum Ingestion CLI.

Unified ingestion for MIT curriculum data including:
- Mission texts (MIT, EECS)
- Course catalog descriptions
- OCW course materials

Usage:
    python -m ads_core.ingest.mit --manifest data/manifests/mit_sources.yaml --out data/processed/mit/artifacts.jsonl

License considerations:
- MIT Catalog: Public information for educational purposes
- MIT OCW: CC BY-NC-SA 4.0
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml

from ads_core.data.schemas import Artifact
from ads_core.ingest.mit_catalog import MitCatalogConnector
from ads_core.ingest.mit_ocw import MitOcwConnector, create_mit_fixtures


def load_manifest(manifest_path: Path) -> dict:
    """Load and validate manifest file."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    # Validate required fields
    required = ["version", "institution"]
    for field in required:
        if field not in manifest:
            raise ValueError(f"Manifest missing required field: {field}")

    return manifest


def write_artifacts_jsonl(artifacts: List[Artifact], output_path: Path) -> int:
    """Write artifacts to JSONL file.

    Args:
        artifacts: List of artifacts to write
        output_path: Output file path

    Returns:
        Number of artifacts written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for artifact in artifacts:
            # Convert to dict, handling datetime
            data = artifact.model_dump()
            if data.get("timestamp"):
                data["timestamp"] = data["timestamp"].isoformat()
            f.write(json.dumps(data) + "\n")
            count += 1

    return count


def write_provenance_jsonl(artifacts: List[Artifact], output_path: Path) -> None:
    """Write provenance records to separate JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for artifact in artifacts:
            prov = artifact.metadata.get("provenance", {})
            record = {
                "artifact_id": artifact.artifact_id,
                "source_url": artifact.source_url,
                "source_hash": artifact.source_hash,
                "timestamp": artifact.timestamp.isoformat() if artifact.timestamp else None,
                "license": artifact.metadata.get("license", ""),
                **prov,
            }
            f.write(json.dumps(record) + "\n")


def run_ingestion(
    manifest_path: Path,
    output_dir: Path,
    snapshot_dir: Optional[Path] = None,
    max_courses: Optional[int] = None,
    live_fetch: bool = False,
    verbose: bool = False,
) -> dict:
    """Run MIT curriculum ingestion.

    Args:
        manifest_path: Path to mit_sources.yaml
        output_dir: Output directory for artifacts
        snapshot_dir: Optional snapshot directory
        max_courses: Limit courses (for testing)
        live_fetch: Enable live web fetching
        verbose: Print progress

    Returns:
        Summary dict with counts and paths
    """
    manifest = load_manifest(manifest_path)

    if verbose:
        print(f"Loaded manifest: {manifest.get('institution')} v{manifest.get('version')}")

    all_artifacts: List[Artifact] = []

    # Get department IDs from manifest
    dept_ids = []
    for dept in manifest.get("departments", {}).get("primary", []):
        dept_ids.append(dept.get("id"))
    for dept in manifest.get("departments", {}).get("secondary", []):
        dept_ids.append(dept.get("id"))

    if not dept_ids:
        dept_ids = ["6"]  # Default to EECS

    # 1. Ingest from MIT Catalog
    if verbose:
        print(f"Ingesting MIT Catalog for departments: {dept_ids}")

    catalog_connector = MitCatalogConnector(
        snapshot_dir=snapshot_dir / "mit_catalog" if snapshot_dir else None,
        departments=dept_ids,
        max_courses=max_courses,
        live_fetch=live_fetch,
    )

    catalog_count = 0
    for artifact in catalog_connector.fetch():
        all_artifacts.append(artifact)
        catalog_count += 1

    if verbose:
        print(f"  Catalog courses: {catalog_count}")

    # 2. Ingest from OCW (with missions)
    if verbose:
        print("Ingesting MIT OCW courses and missions")

    ocw_connector = MitOcwConnector(
        manifest_path=manifest_path,
        snapshot_dir=snapshot_dir / "ocw" if snapshot_dir else None,
        max_courses=max_courses,
        live_fetch=live_fetch,
        include_missions=True,
    )

    mission_count = 0
    ocw_count = 0
    for artifact in ocw_connector.fetch():
        # Check for duplicates
        existing_ids = {a.artifact_id for a in all_artifacts}
        if artifact.artifact_id not in existing_ids:
            all_artifacts.append(artifact)
            if artifact.type.value == "MISSION":
                mission_count += 1
            else:
                ocw_count += 1

    if verbose:
        print(f"  Missions: {mission_count}")
        print(f"  OCW courses: {ocw_count}")

    # Write outputs
    artifacts_path = output_dir / "artifacts.jsonl"
    provenance_path = output_dir / "provenance.jsonl"

    artifact_count = write_artifacts_jsonl(all_artifacts, artifacts_path)
    write_provenance_jsonl(all_artifacts, provenance_path)

    # Write manifest snapshot
    manifest_snapshot_path = output_dir / "manifest_snapshot.yaml"
    manifest_snapshot = {
        **manifest,
        "ingestion_run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "artifact_count": artifact_count,
            "catalog_count": catalog_count,
            "ocw_count": ocw_count,
            "mission_count": mission_count,
            "live_fetch": live_fetch,
            "max_courses": max_courses,
        },
    }
    manifest_snapshot_path.write_text(
        yaml.dump(manifest_snapshot, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    summary = {
        "total_artifacts": artifact_count,
        "catalog_courses": catalog_count,
        "ocw_courses": ocw_count,
        "missions": mission_count,
        "artifacts_path": str(artifacts_path),
        "provenance_path": str(provenance_path),
        "manifest_snapshot_path": str(manifest_snapshot_path),
    }

    if verbose:
        print(f"\nTotal artifacts: {artifact_count}")
        print(f"Written to: {artifacts_path}")

    return summary


def ensure_fixtures(fixture_dir: Path, verbose: bool = False) -> None:
    """Ensure test fixtures exist."""
    if verbose:
        print(f"Ensuring fixtures in: {fixture_dir}")

    create_mit_fixtures(fixture_dir)

    if verbose:
        print("  Fixtures created/verified")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MIT Curriculum Ingestion for ADS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using fixtures (default, reproducible)
  python -m ads_core.ingest.mit --manifest data/manifests/mit_sources.yaml

  # With live fetching
  python -m ads_core.ingest.mit --manifest data/manifests/mit_sources.yaml --live

  # Limited courses for testing
  python -m ads_core.ingest.mit --manifest data/manifests/mit_sources.yaml --max-courses 5

  # Create fixtures only
  python -m ads_core.ingest.mit --create-fixtures
        """,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/mit_sources.yaml"),
        help="Path to manifest YAML file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/mit"),
        help="Output directory",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Directory with pre-downloaded data (uses fixtures if not specified)",
    )
    parser.add_argument(
        "--max-courses",
        type=int,
        default=None,
        help="Maximum courses to process (for testing)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live web fetching (default: use fixtures/snapshots)",
    )
    parser.add_argument(
        "--create-fixtures",
        action="store_true",
        help="Create test fixtures and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Set default fixture directory
    fixture_dir = Path(__file__).parent.parent.parent.parent / "data" / "fixtures"

    # Create fixtures if requested
    if args.create_fixtures:
        ensure_fixtures(fixture_dir, verbose=args.verbose)
        print(f"Fixtures created at: {fixture_dir}")
        return 0

    # Ensure fixtures exist
    ensure_fixtures(fixture_dir, verbose=args.verbose)

    # Use fixtures as snapshot if no snapshot dir specified
    snapshot_dir = args.snapshot_dir or fixture_dir

    try:
        summary = run_ingestion(
            manifest_path=args.manifest,
            output_dir=args.out,
            snapshot_dir=snapshot_dir,
            max_courses=args.max_courses,
            live_fetch=args.live,
            verbose=args.verbose,
        )

        if args.verbose:
            print("\nSummary:")
            for k, v in summary.items():
                print(f"  {k}: {v}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
