"""UC Berkeley Outcomes Ingestion CLI.

Ingests First Destination Survey (FDS) career outcomes data from UC Berkeley.
Designed to work with manually exported CSV snapshots for reproducibility.

Usage:
    python -m ads_core.ingest.ucb --manifest data/manifests/ucb_sources.yaml --out_dir data/processed/ucb

Data sources:
- First Destination Survey: https://opa.berkeley.edu/campus-surveys/survey-results-reporting-analysis/first-destination-survey
- Where Do Cal Grads Go?: https://career.berkeley.edu/start-exploring/where-do-cal-grads-go/

License: Public aggregate data for educational research
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
from ads_core.ingest.ucb_outcomes import (
    UcbOutcomesConnector,
    create_ucb_fixture,
    export_outcomes_csv,
)
from ads_core.utils.hashing import sha256_text


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
    """Run UCB outcomes ingestion.

    Args:
        manifest_path: Path to ucb_sources.yaml
        output_dir: Output directory for artifacts
        raw_dir: Directory with raw CSV files (uses fixtures if not provided)
        max_records: Limit records (for testing)
        verbose: Print progress

    Returns:
        Summary dict with counts and paths
    """
    manifest = load_manifest(manifest_path)

    if verbose:
        print(f"Loaded manifest: {manifest.get('institution')} v{manifest.get('version')}")

    # Ensure fixtures exist
    fixture_dir = Path(__file__).parent.parent.parent.parent / "data" / "fixtures" / "ucb"
    if not fixture_dir.exists() or not list(fixture_dir.glob("*.csv")):
        if verbose:
            print(f"Creating fixtures at: {fixture_dir}")
        create_ucb_fixture(fixture_dir)

    # Use raw_dir or fall back to fixtures
    data_dir = raw_dir or fixture_dir

    if verbose:
        print(f"Loading outcomes from: {data_dir}")

    # Initialize connector
    connector = UcbOutcomesConnector(
        manifest_path=manifest_path,
        raw_dir=data_dir,
        max_records=max_records,
    )

    # Load records and export CSV
    records = connector.load_records()
    outcomes_csv_path = output_dir / "outcomes_major_level.csv"
    csv_count = export_outcomes_csv(records, outcomes_csv_path)

    if verbose:
        print(f"  Outcome records: {csv_count}")

    # Generate artifacts
    all_artifacts: List[Artifact] = list(connector.fetch())

    if verbose:
        print(f"  Artifacts generated: {len(all_artifacts)}")

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
            "outcome_records": csv_count,
            "data_dir": str(data_dir),
        },
    }
    manifest_snapshot_path.write_text(
        yaml.dump(manifest_snapshot, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    # Compute hash of outcomes CSV for reproducibility
    csv_content = outcomes_csv_path.read_text(encoding="utf-8")
    csv_hash = sha256_text(csv_content)

    summary = {
        "total_artifacts": artifact_count,
        "outcome_records": csv_count,
        "outcomes_csv_path": str(outcomes_csv_path),
        "outcomes_csv_hash": csv_hash,
        "artifacts_path": str(artifacts_path),
        "provenance_path": str(provenance_path),
        "manifest_snapshot_path": str(manifest_snapshot_path),
    }

    if verbose:
        print(f"\nTotal artifacts: {artifact_count}")
        print(f"Outcomes CSV: {outcomes_csv_path}")
        print(f"Outcomes CSV hash: {csv_hash[:16]}...")

    return summary


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="UC Berkeley Outcomes Ingestion for ADS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using fixtures (default, reproducible)
  python -m ads_core.ingest.ucb --manifest data/manifests/ucb_sources.yaml --out_dir data/processed/ucb

  # With custom raw data directory
  python -m ads_core.ingest.ucb --manifest data/manifests/ucb_sources.yaml --raw_dir data/raw/ucb/fds --out_dir data/processed/ucb

  # Limited records for testing
  python -m ads_core.ingest.ucb --manifest data/manifests/ucb_sources.yaml --max-records 5

  # Create fixtures only
  python -m ads_core.ingest.ucb --create-fixtures
        """,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/ucb_sources.yaml"),
        help="Path to manifest YAML file",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/processed/ucb"),
        help="Output directory",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=None,
        help="Directory with raw CSV files (uses fixtures if not specified)",
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
        fixture_dir = Path(__file__).parent.parent.parent.parent / "data" / "fixtures" / "ucb"
        create_ucb_fixture(fixture_dir)
        print(f"Fixtures created at: {fixture_dir}")
        return 0

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
