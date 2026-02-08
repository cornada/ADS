#!/usr/bin/env python3
"""ADS-Unified-v4 Dataset Integrity Validator.

Checks file presence, record counts, column schemas, hash integrity,
and scans for potential PII leakage.

Usage:
    python scripts/validate_dataset.py [--root /path/to/ads-benchmark-v4]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ── Expected file manifest with SHA-256 hashes ─────────────────────────
EXPECTED_FILES = {
    "courses.csv": "8f246faa12bc7593e94e0f6a55ca98e44d54dd52f5fa6cb166343b75737cc624",
    "artifacts.jsonl": "eeaed5051a82cf15424ba81406215fb8bddd675dca8c93f9dd4d116ab0b493e0",
    "programs.csv": "badafaa704d3b7b4bbc11bd2a7b322e241bf4491e3c34bffa1891d31d2221575",
    "onet/occupations.csv": "91deed7c08f1e530d29defa46337465a8aa5b91896a0e8251aa9bbd28eeecd62",
    "onet/skills.csv": "0504a85c6e66e35a640a63cfe6f785e32b7c7cd5452984aa6e0ab62ad3f3ec62",
    "onet/tasks.csv": "1fb6f18578961fa9fa0f821ffe299646423b2d673966c4bfb531c6f6cf3e9fa7",
    "onet/technology.csv": "d0b995a1d4d1ced583dee2300fe3024f95262cd8cada8649e68dc00899662fd3",
    "onet/time_travel/onet_26_3.csv": "2dfb672890e76ec5cbdd79fe81af750865d0909a9c5411e660987622373cc972",
    "onet/time_travel/onet_27_3.csv": "b9d06984e5576ee9d8ae63ddcbc6c8a1551962d57ec5d54276794791869b9016",
    "onet/time_travel/onet_28_1.csv": "70febc46867708037635935fc810da196ff6e154895572821379e1aa8f672e3f",
    "onet/time_travel/onet_30_0.csv": "f2033115bba52226b23beab958b869ac50ca654dcf24f2ea397c6392691484f5",
    "onet/time_travel/PACK_META.csv": "de29fb57137fa5f92089d41e088ae2557544f94e6d7325ad6b088f4a38c7452d",
    "crosswalks/cip_to_soc.csv": "882d9a19205037dbecfb9d6c11a7fc12eddebf9e57721b14c39fdc9318ac4ce1",
    "crosswalks/soc_to_cip.csv": "0c8ddbf574e8af4c3259a7c9da077c746f2105d4c0520a751ed7116ab81f4a71",
    "missions/mission_corpus.csv": "edbd3bc0ea310ad68cc6dc5dacf0795470d0f23362ae64fd8108fedd84c9c474",
    "results/pareto_minilm.json": "8b4cb2b04638b81163f1f4b737a9dde06656755cc525c34eeca2acd6c9b78208",
    "results/pareto_mpnet.json": "805952207ba05777710bf5abe1193e9da64497bef2b1abcfb85bb75f19e9c372",
    "results/pareto_learned_lens.json": "b4a97babb062c0327931c23202a2f3a37da0f3722380b97b607e258a7abfa157",
    "results/baselines.json": "c2cadee71629b87963f65529194deec50e2f696d31a06ad8d32842e4284a509c",
    "results/goal_sensitivity.json": "254780768e26b06bee2cb841212cbd27002037ea80a76eae4ea0ef9c34a21410",
}

# ── Expected record counts ──────────────────────────────────────────────
EXPECTED_COUNTS = {
    "courses": 27586,
    "artifacts": 28633,
    "artifact_types": {"COURSE": 27555, "JOB_ROLE": 1016, "MISSION": 62},
    "programs": 398,
    "occupations": 1016,
    "skills": 71520,
    "tasks": 13326,
    "technology": 32773,
    "missions": 62,
}

COURSES_COLUMNS = {"course_id", "course_name", "description", "units",
                   "department", "level", "prerequisites", "university"}
EXPECTED_UNIVERSITIES = {"MIT", "UC Berkeley", "Stanford", "UIUC",
                         "Cornell", "KTH", "Edinburgh"}

# ── PII patterns ────────────────────────────────────────────────────────
PII_PATTERNS = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
    (r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", "SSN-like"),
    (r"\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b", "phone"),
    (r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "phone"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def check_files(root: Path) -> list[str]:
    """Verify all expected files exist and match SHA-256 hashes."""
    errors = []
    for relpath, expected_hash in EXPECTED_FILES.items():
        fpath = root / relpath
        if not fpath.exists():
            errors.append(f"MISSING: {relpath}")
            continue
        actual = sha256_file(fpath)
        if actual != expected_hash:
            errors.append(f"HASH MISMATCH: {relpath} (expected {expected_hash[:16]}..., got {actual[:16]}...)")
    return errors


def check_courses(root: Path) -> list[str]:
    """Validate courses.csv schema and counts."""
    errors = []
    fpath = root / "courses.csv"
    if not fpath.exists():
        return ["courses.csv not found"]

    with open(fpath, newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing_cols = COURSES_COLUMNS - columns
        if missing_cols:
            errors.append(f"courses.csv missing columns: {missing_cols}")

        rows = list(reader)

    n = len(rows)
    if n != EXPECTED_COUNTS["courses"]:
        errors.append(f"courses.csv: expected {EXPECTED_COUNTS['courses']} rows, got {n}")

    # Check uniqueness
    ids = [r["course_id"] for r in rows]
    if len(ids) != len(set(ids)):
        dupes = len(ids) - len(set(ids))
        errors.append(f"courses.csv: {dupes} duplicate course_id values")

    # Check universities
    unis = set(r["university"] for r in rows)
    missing_unis = EXPECTED_UNIVERSITIES - unis
    if missing_unis:
        errors.append(f"courses.csv missing universities: {missing_unis}")

    return errors


def check_artifacts(root: Path) -> list[str]:
    """Validate artifacts.jsonl counts and type distribution."""
    errors = []
    fpath = root / "artifacts.jsonl"
    if not fpath.exists():
        return ["artifacts.jsonl not found"]

    types: Counter[str] = Counter()
    n = 0
    with open(fpath) as f:
        for line in f:
            obj = json.loads(line)
            types[obj.get("type", "UNKNOWN")] += 1
            n += 1

    if n != EXPECTED_COUNTS["artifacts"]:
        errors.append(f"artifacts.jsonl: expected {EXPECTED_COUNTS['artifacts']} records, got {n}")

    for atype, expected in EXPECTED_COUNTS["artifact_types"].items():
        actual = types.get(atype, 0)
        if actual != expected:
            errors.append(f"artifacts.jsonl: expected {expected} {atype}, got {actual}")

    return errors


def check_csv_count(root: Path, relpath: str, key: str) -> list[str]:
    """Check a CSV file has the expected number of data rows."""
    errors = []
    fpath = root / relpath
    if not fpath.exists():
        return [f"{relpath} not found"]
    with open(fpath, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        n = sum(1 for _ in reader)
    expected = EXPECTED_COUNTS[key]
    if n != expected:
        errors.append(f"{relpath}: expected {expected} rows, got {n}")
    return errors


def check_pii(root: Path) -> list[str]:
    """Scan text fields for potential PII patterns."""
    warnings = []
    compiled = [(re.compile(p), name) for p, name in PII_PATTERNS]

    # Check courses.csv descriptions
    courses_path = root / "courses.csv"
    if courses_path.exists():
        with open(courses_path, newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                text = row.get("description", "")
                for pat, name in compiled:
                    matches = pat.findall(text)
                    if matches:
                        warnings.append(
                            f"PII ({name}) in courses.csv row {i+1}: "
                            f"{matches[0][:30]}..."
                        )

    # Check artifacts.jsonl text
    artifacts_path = root / "artifacts.jsonl"
    if artifacts_path.exists():
        with open(artifacts_path) as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                text = obj.get("text", "")
                for pat, name in compiled:
                    matches = pat.findall(text)
                    if matches:
                        warnings.append(
                            f"PII ({name}) in artifacts.jsonl line {i+1}: "
                            f"{matches[0][:30]}..."
                        )

    return warnings


def check_croissant(root: Path) -> list[str]:
    """Verify croissant.json stats match actual data."""
    errors = []
    fpath = root / "croissant.json"
    if not fpath.exists():
        return ["croissant.json not found"]

    with open(fpath) as f:
        meta = json.load(f)

    stats = meta.get("ml:datasetStats", {})
    if stats.get("numCourses") != EXPECTED_COUNTS["courses"]:
        errors.append(
            f"croissant.json numCourses: {stats.get('numCourses')} "
            f"!= expected {EXPECTED_COUNTS['courses']}"
        )
    if stats.get("numArtifactsTotal") != EXPECTED_COUNTS["artifacts"]:
        errors.append(
            f"croissant.json numArtifactsTotal: {stats.get('numArtifactsTotal')} "
            f"!= expected {EXPECTED_COUNTS['artifacts']}"
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ADS-Unified-v4 dataset")
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to ads-benchmark-v4 directory",
    )
    args = parser.parse_args()
    root = args.root

    print(f"Validating dataset at: {root}\n")

    all_errors: list[str] = []
    all_warnings: list[str] = []

    # 1. File integrity
    print("Checking file integrity...")
    errs = check_files(root)
    all_errors.extend(errs)
    print(f"  {len(EXPECTED_FILES)} files checked, {len(errs)} errors")

    # 2. Courses
    print("Checking courses.csv...")
    errs = check_courses(root)
    all_errors.extend(errs)
    print(f"  {len(errs)} errors")

    # 3. Artifacts
    print("Checking artifacts.jsonl...")
    errs = check_artifacts(root)
    all_errors.extend(errs)
    print(f"  {len(errs)} errors")

    # 4. O*NET counts
    print("Checking O*NET files...")
    for relpath, key in [
        ("onet/occupations.csv", "occupations"),
        ("onet/skills.csv", "skills"),
        ("onet/tasks.csv", "tasks"),
        ("onet/technology.csv", "technology"),
        ("missions/mission_corpus.csv", "missions"),
        ("programs.csv", "programs"),
    ]:
        errs = check_csv_count(root, relpath, key)
        all_errors.extend(errs)
    print(f"  6 files checked")

    # 5. Croissant metadata consistency
    print("Checking croissant.json consistency...")
    errs = check_croissant(root)
    all_errors.extend(errs)
    print(f"  {len(errs)} errors")

    # 6. PII scan
    print("Scanning for PII...")
    warns = check_pii(root)
    all_warnings.extend(warns)
    print(f"  {len(warns)} warnings")

    # Summary
    print(f"\n{'='*60}")
    if all_errors:
        print(f"ERRORS ({len(all_errors)}):")
        for e in all_errors:
            print(f"  [ERROR] {e}")
    if all_warnings:
        print(f"\nWARNINGS ({len(all_warnings)}):")
        for w in all_warnings[:20]:
            print(f"  [WARN]  {w}")
        if len(all_warnings) > 20:
            print(f"  ... and {len(all_warnings) - 20} more")

    if not all_errors and not all_warnings:
        print("ALL CHECKS PASSED")
        return 0
    elif not all_errors:
        print(f"\nPASSED with {len(all_warnings)} warnings")
        return 0
    else:
        print(f"\nFAILED: {len(all_errors)} errors, {len(all_warnings)} warnings")
        return 1


if __name__ == "__main__":
    sys.exit(main())
