"""CLI for running data ingestion pipelines.

Usage:
    python -m experiments.ingest --config experiments/conf/dataset/open_data.yaml
    python -m experiments.ingest --source onet --output data/processed/onet
    python -m experiments.ingest --source ocw --output data/processed/ocw
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Python 3.14 compatibility fix for Hydra's LazyCompletionHelp
if sys.version_info >= (3, 14):
    import argparse as _argparse

    _original_check_help = _argparse.ArgumentParser._check_help

    def _patched_check_help(self, action):
        if action.help is not None and not isinstance(action.help, str):
            return
        return _original_check_help(self, action)

    _argparse.ArgumentParser._check_help = _patched_check_help

from ads_core.data.schemas import Artifact
from ads_core.data.storage import LocalArtifactStore
from ads_core.ingest.base import BaseConnector
from ads_core.ingest.onet import OnetConnector, create_onet_fixture
from ads_core.ingest.ocw import OcwConnector, create_ocw_fixture
from ads_core.utils.hashing import sha256_text


def get_connector(source: str, **kwargs) -> BaseConnector:
    """Get connector instance by source name."""
    connectors = {
        "onet": OnetConnector,
        "ocw": OcwConnector,
    }
    if source not in connectors:
        raise ValueError(f"Unknown source: {source}. Available: {list(connectors.keys())}")
    return connectors[source](**kwargs)


def run_ingestion(
    sources: List[str],
    output_dir: Path,
    max_items: Optional[int] = None,
    create_fixtures: bool = False,
) -> dict:
    """Run ingestion from multiple sources and save to output directory.

    Args:
        sources: List of source names (e.g., ["onet", "ocw"])
        output_dir: Directory to save artifacts
        max_items: Max items per source (for testing)
        create_fixtures: If True, create fixture data first

    Returns:
        Summary dict with counts and paths
    """
    output_dir = Path(output_dir)
    store = LocalArtifactStore(output_dir / "store")

    # Optionally create fixtures first
    if create_fixtures:
        fixtures_base = Path(__file__).parent.parent / "data" / "fixtures"
        if "onet" in sources:
            create_onet_fixture(fixtures_base / "onet")
            print(f"[OK] Created O*NET fixture at {fixtures_base / 'onet'}")
        if "ocw" in sources:
            create_ocw_fixture(fixtures_base / "ocw")
            print(f"[OK] Created OCW fixture at {fixtures_base / 'ocw'}")

    all_artifacts: List[Artifact] = []
    source_counts = {}

    for source in sources:
        print(f"[..] Ingesting from {source}...")
        try:
            # Each connector has its own limit parameter name
            if source == "onet":
                connector = get_connector(source, max_occupations=max_items)
            elif source == "ocw":
                connector = get_connector(source, max_courses=max_items)
            else:
                connector = get_connector(source)
            artifacts = connector.fetch_all()
            source_counts[source] = len(artifacts)
            all_artifacts.extend(artifacts)
            print(f"[OK] {source}: {len(artifacts)} artifacts")
        except FileNotFoundError as e:
            print(f"[WARN] {source}: {e}")
            source_counts[source] = 0

    # Save artifacts
    if all_artifacts:
        store.save_artifacts(all_artifacts)

    # Write provenance summary
    provenance_summary = {
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "sources": source_counts,
        "total_artifacts": len(all_artifacts),
        "artifact_types": {},
        "output_path": str(store.artifacts_path()),
    }

    for art in all_artifacts:
        art_type = art.type.value
        provenance_summary["artifact_types"][art_type] = (
            provenance_summary["artifact_types"].get(art_type, 0) + 1
        )

    # Compute dataset hash for reproducibility
    if all_artifacts:
        all_text = "\n".join(sorted(a.artifact_id for a in all_artifacts))
        provenance_summary["dataset_hash"] = sha256_text(all_text)[:16]

    store.write_meta(provenance_summary)
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance_summary, indent=2), encoding="utf-8"
    )

    return provenance_summary


def main():
    parser = argparse.ArgumentParser(
        description="Ingest data from open sources into ADS artifact format."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["onet", "ocw"],
        help="Data sources to ingest from (default: onet ocw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for artifacts (default: data/processed)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Max items per source (for testing)",
    )
    parser.add_argument(
        "--create-fixtures",
        action="store_true",
        help="Create fixture data files before ingestion",
    )

    args = parser.parse_args()

    summary = run_ingestion(
        sources=args.sources,
        output_dir=args.output,
        max_items=args.max_items,
        create_fixtures=args.create_fixtures,
    )

    print("\n[OK] Ingestion complete:")
    print(f"     Total artifacts: {summary['total_artifacts']}")
    print(f"     Output: {summary['output_path']}")
    if summary.get("dataset_hash"):
        print(f"     Dataset hash: {summary['dataset_hash']}")


if __name__ == "__main__":
    main()
