"""ASU Ingestion CLI.

Unified ingestion for ASU data including:
- First Destination Survey (FDS) schema/instrument
- Career outcomes reports

Usage:
    python -m ads_core.ingest.asu --manifest data/manifests/asu_sources.yaml --out_dir data/processed/asu

Data sources:
- FDS Survey Preview: https://uoeee.asu.edu/sites/g/files/litvpz631/files/docs/Survey/FDS%20Survey%20Preview.pdf
- Career Outcomes: https://provost.asu.edu/sites/g/files/litvpz671/files/page/2550/career_outcomes_1516.pdf
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
from ads_core.ingest.asu_fds import AsuFdsConnector, create_fds_fixture
from ads_core.ingest.asu_outcomes import (
    AsuOutcomesConnector,
    create_outcomes_fixture,
    export_outcomes_csv,
)
from ads_core.utils.hashing import sha256_text


def load_manifest(manifest_path: Path) -> dict:
    """Load and validate manifest file."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    required = ["version", "institution"]
    for field in required:
        if field not in manifest:
            raise ValueError(f"Manifest missing required field: {field}")

    return manifest


def write_artifacts_jsonl(artifacts: List[Artifact], output_path: Path) -> int:
    """Write artifacts to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for artifact in artifacts:
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
                "license": artifact.metadata.get("license", "unknown"),
                **prov,
            }
            f.write(json.dumps(record) + "\n")


def run_ingestion(
    manifest_path: Path,
    output_dir: Path,
    raw_dir: Optional[Path] = None,
    max_records: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Run ASU ingestion.

    Args:
        manifest_path: Path to asu_sources.yaml
        output_dir: Output directory for artifacts
        raw_dir: Directory with raw PDF/CSV files
        max_records: Limit records (for testing)
        verbose: Print progress

    Returns:
        Summary dict with counts and paths
    """
    manifest = load_manifest(manifest_path)

    if verbose:
        print(f"Loaded manifest: {manifest.get('institution')} v{manifest.get('version')}")

    # Ensure fixtures exist
    fixture_dir = Path(__file__).parent.parent.parent.parent / "data" / "fixtures" / "asu"
    if not fixture_dir.exists():
        if verbose:
            print(f"Creating fixtures at: {fixture_dir}")
        create_fds_fixture(fixture_dir)
        create_outcomes_fixture(fixture_dir)

    all_artifacts: List[Artifact] = []

    # 1. Ingest FDS schema
    if verbose:
        print("Ingesting FDS survey schema...")

    fds_connector = AsuFdsConnector(
        manifest_path=manifest_path,
        raw_dir=raw_dir,
    )

    fds_count = 0
    schema = fds_connector.load_schema()
    for artifact in fds_connector.fetch():
        all_artifacts.append(artifact)
        fds_count += 1

    if verbose:
        print(f"  FDS schema artifacts: {fds_count}")

    # Write FDS schema JSON
    fds_schema_path = output_dir / "fds_schema.json"
    fds_schema_path.parent.mkdir(parents=True, exist_ok=True)
    fds_schema_path.write_text(
        json.dumps(schema.to_dict(), indent=2),
        encoding="utf-8"
    )

    # 2. Ingest outcomes
    if verbose:
        print("Ingesting career outcomes...")

    outcomes_connector = AsuOutcomesConnector(
        manifest_path=manifest_path,
        raw_dir=raw_dir,
        max_records=max_records,
    )

    records = outcomes_connector.load_records()
    outcomes_csv_path = output_dir / "outcomes_summary.csv"
    csv_count = export_outcomes_csv(records, outcomes_csv_path)

    outcomes_count = 0
    for artifact in outcomes_connector.fetch():
        all_artifacts.append(artifact)
        outcomes_count += 1

    if verbose:
        print(f"  Outcomes records: {csv_count}")
        print(f"  Outcomes artifacts: {outcomes_count}")

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
            "fds_artifacts": fds_count,
            "outcomes_artifacts": outcomes_count,
            "outcomes_records": csv_count,
        },
    }
    manifest_snapshot_path.write_text(
        yaml.dump(manifest_snapshot, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    # Compute hashes
    schema_hash = sha256_text(fds_schema_path.read_text(encoding="utf-8"))
    csv_hash = sha256_text(outcomes_csv_path.read_text(encoding="utf-8"))

    summary = {
        "total_artifacts": artifact_count,
        "fds_artifacts": fds_count,
        "fds_sections": len(schema.sections),
        "fds_questions": sum(len(s.questions) for s in schema.sections),
        "outcomes_artifacts": outcomes_count,
        "outcomes_records": csv_count,
        "fds_schema_path": str(fds_schema_path),
        "fds_schema_hash": schema_hash,
        "outcomes_csv_path": str(outcomes_csv_path),
        "outcomes_csv_hash": csv_hash,
        "artifacts_path": str(artifacts_path),
        "provenance_path": str(provenance_path),
    }

    if verbose:
        print(f"\nTotal artifacts: {artifact_count}")
        print(f"FDS schema: {fds_schema_path}")
        print(f"Outcomes CSV: {outcomes_csv_path}")

    return summary


def ensure_fixtures(fixture_dir: Path, verbose: bool = False) -> None:
    """Ensure test fixtures exist."""
    if verbose:
        print(f"Ensuring fixtures in: {fixture_dir}")

    if not (fixture_dir / "fds_schema.json").exists():
        create_fds_fixture(fixture_dir)
        if verbose:
            print("  Created FDS schema fixture")

    if not (fixture_dir / "outcomes_summary.csv").exists():
        create_outcomes_fixture(fixture_dir)
        if verbose:
            print("  Created outcomes fixture")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ASU Ingestion for ADS (FDS + Outcomes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using fixtures (default, reproducible)
  python -m ads_core.ingest.asu --manifest data/manifests/asu_sources.yaml --out_dir data/processed/asu

  # With custom raw data directory
  python -m ads_core.ingest.asu --manifest data/manifests/asu_sources.yaml --raw_dir data/raw/asu --out_dir data/processed/asu

  # Create fixtures only
  python -m ads_core.ingest.asu --create-fixtures
        """,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/asu_sources.yaml"),
        help="Path to manifest YAML file",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/processed/asu"),
        help="Output directory",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=None,
        help="Directory with raw PDF/CSV files",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum records to process (for testing)",
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

    # Create fixtures if requested
    if args.create_fixtures:
        fixture_dir = Path(__file__).parent.parent.parent.parent / "data" / "fixtures" / "asu"
        ensure_fixtures(fixture_dir, verbose=True)
        print(f"Fixtures created at: {fixture_dir}")
        return 0

    # Ensure fixtures exist
    fixture_dir = Path(__file__).parent.parent.parent.parent / "data" / "fixtures" / "asu"
    ensure_fixtures(fixture_dir, verbose=args.verbose)

    try:
        summary = run_ingestion(
            manifest_path=args.manifest,
            output_dir=args.out_dir,
            raw_dir=args.raw_dir,
            max_records=args.max_records,
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
