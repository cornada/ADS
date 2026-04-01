"""Generate COMPETENCY artifacts from MISIS competency mappings.

Reads data/raw/misis/competencies.csv and produces unique competency
artifacts for the Pareto optimizer's 5th objective.

Each FGOS competency (УК/ОПК/ПК) becomes an artifact whose text describes
the regulatory requirement. The optimizer then measures how well each
course in a curriculum plan covers the full set of competencies.

Usage:
    python -m ads_core.ingest.competencies
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_COMPETENCIES = _REPO_ROOT / "data" / "raw" / "misis" / "competencies.csv"
_DEFAULT_OUT = _REPO_ROOT / "data" / "raw" / "misis" / "competency_artifacts.jsonl"

# Competency type descriptions for richer text
_TYPE_DESC = {
    "УК": "Universal competency (Универсальная компетенция)",
    "ОПК": "General professional competency (Общепрофессиональная компетенция)",
    "ПК": "Professional competency (Профессиональная компетенция)",
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def build_competency_artifacts(
    competencies_path: Path,
    *,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Build unique competency artifacts from course-competency mappings.

    Groups by (competency_code, fgos_code) to create one artifact per
    unique competency within each FGOS direction. Uses the longest
    competency_text as the canonical description.
    """
    # Group: (code, fgos) → list of records
    groups: Dict[str, List[Dict]] = defaultdict(list)

    with competencies_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("competency_code", "").strip()
            fgos = row.get("fgos_code", "").strip()
            if not code:
                continue
            key = f"{code}|{fgos}" if fgos else code
            groups[key].append(row)

    now = datetime.now(timezone.utc).isoformat()
    artifacts = []

    for key, records in groups.items():
        code = records[0]["competency_code"].strip()
        comp_type = records[0].get("competency_type", "").strip()
        fgos = records[0].get("fgos_code", "").strip()

        # Use shortest meaningful text (less noise from concatenated blocks)
        texts = [_norm(r.get("competency_text", "")) for r in records if r.get("competency_text")]
        # Filter to reasonable length (some are very long concatenated blocks)
        clean_texts = [t for t in texts if 20 < len(t) < 500]
        if clean_texts:
            canonical = min(clean_texts, key=len)
        elif texts:
            canonical = texts[0][:500]
        else:
            canonical = f"Competency {code}"

        type_desc = _TYPE_DESC.get(comp_type, comp_type)
        course_count = len(records)
        courses = sorted(set(r.get("course_name", "") for r in records if r.get("course_name")))

        parts = [
            f"FGOS Competency {code}.",
            f"Type: {type_desc}.",
            f"FGOS direction: {fgos}." if fgos else "",
            f"Description: {canonical}",
            f"Required in {course_count} courses.",
        ]
        text = _norm(" ".join(p for p in parts if p))

        hid = _sha256_text(f"{code}|{fgos}")[:12]
        aid = f"competency:MISIS:{hid}"

        rec = {
            "artifact_id": aid,
            "type": "COMPETENCY",
            "source_url": f"urn:ads:misis:competency:{code}:{fgos}",
            "source_hash": _sha256_text(text),
            "timestamp": now,
            "text": text,
            "metadata": {
                "institution": "MISIS",
                "competency_code": code,
                "competency_type": comp_type,
                "fgos_code": fgos,
                "course_count": course_count,
                "sample_courses": courses[:5],
                "country": "RU",
                "provenance": {
                    "source_url": str(competencies_path),
                    "fetched_at": now,
                    "raw_text_hash": _sha256_text(text),
                    "dataset": "misis",
                },
            },
        }
        artifacts.append(rec)

    if verbose:
        print(f"  Competency artifacts: {len(artifacts)}")
        by_type = defaultdict(int)
        for a in artifacts:
            by_type[a["metadata"]["competency_type"]] += 1
        for t, c in sorted(by_type.items()):
            print(f"    {t}: {c}")

    return artifacts


def save_artifacts(
    artifacts: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Save competency artifacts to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for rec in artifacts:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Wrote %d competency artifacts to %s", len(artifacts), output_path)
    return output_path


def run_ingestion(
    competencies_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    *,
    verbose: bool = True,
) -> Path:
    """Generate competency artifacts.

    Returns path to output JSONL.
    """
    if competencies_path is None:
        competencies_path = _DEFAULT_COMPETENCIES
    if output_path is None:
        output_path = _DEFAULT_OUT

    if verbose:
        print("MISIS Competency Artifact Generation")

    artifacts = build_competency_artifacts(competencies_path, verbose=verbose)
    out = save_artifacts(artifacts, output_path)

    if verbose:
        print(f"  Output: {out}")
        print("  [OK] Competency artifacts generated")

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ingestion()
