"""Build unified_v5: 8+ universities + 3 labor markets (US/RU/EU).

Merges:
  - Courses: existing unified_v3 base + KTH + Edinburgh + MISIS
  - Labor market: O*NET (US) + Profstandart (RU) + HeadHunter (RU) + ESCO (EU)
  - Missions: existing mission corpus

Generates a single artifacts.jsonl that combines all sources.

Usage:
    python experiments/scripts/merge_unified_v5.py
    python experiments/scripts/merge_unified_v5.py --skip-artifacts
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]

COURSE_FIELDS = [
    "course_id", "course_name", "description", "units",
    "department", "level", "prerequisites", "university",
]

# Course sources to merge into unified_v5
_BASE_COURSES = _REPO_ROOT / "data" / "processed" / "unified_v3" / "courses.csv"

_NEW_COURSE_SOURCES = [
    _REPO_ROOT / "data" / "raw" / "kth" / "courses.csv",
    _REPO_ROOT / "data" / "raw" / "edinburgh" / "courses.csv",
    _REPO_ROOT / "data" / "raw" / "misis" / "courses.csv",
]

# Labor market artifact sources
_MARKET_ARTIFACT_SOURCES = [
    ("O*NET (US)", _REPO_ROOT / "data" / "processed" / "unified_v3" / "artifacts.jsonl", "onet:"),
    ("Profstandart (RU)", _REPO_ROOT / "data" / "raw" / "profstandart" / "artifacts.jsonl", None),
    ("HeadHunter (RU)", _REPO_ROOT / "data" / "raw" / "headhunter" / "artifacts.jsonl", None),
    ("ESCO (EU)", _REPO_ROOT / "data" / "raw" / "esco" / "artifacts.jsonl", None),
    ("Competencies (MISIS)", _REPO_ROOT / "data" / "raw" / "misis" / "competency_artifacts.jsonl", None),
]

# Mission data
_MISSION_CORPUS = _REPO_ROOT / "data" / "processed" / "unified_v3" / "mission_corpus.csv"
_MISSION_STATEMENTS = _REPO_ROOT / "data" / "processed" / "unified_v3" / "mission_statements.csv"

# Output
_OUT_DIR = _REPO_ROOT / "data" / "processed" / "unified_v5"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def _short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def merge_courses(verbose: bool = True) -> pd.DataFrame:
    """Merge all course CSVs, deduplicating by university."""
    all_rows = []
    unis_seen = set()

    # Base courses
    if _BASE_COURSES.exists():
        with _BASE_COURSES.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
                unis_seen.add(row.get("university", ""))
        if verbose:
            print(f"  Base: {len(all_rows)} courses from {unis_seen}")

    # New sources
    for src in _NEW_COURSE_SOURCES:
        if not src.exists():
            if verbose:
                print(f"  [SKIP] {src.name} not found")
            continue
        with src.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            new_rows = list(reader)
        if not new_rows:
            continue
        uni = new_rows[0].get("university", "unknown")
        if uni in unis_seen:
            if verbose:
                print(f"  [SKIP] {uni} already present")
            continue
        unis_seen.add(uni)
        all_rows.extend(new_rows)
        if verbose:
            print(f"  [ADD] {uni}: {len(new_rows)} courses")

    df = pd.DataFrame(all_rows)
    # Ensure standard columns
    for col in COURSE_FIELDS:
        if col not in df.columns:
            df[col] = ""

    if verbose:
        print(f"\n  Total courses: {len(df)}")
        by_uni = df["university"].value_counts()
        for uni, count in by_uni.items():
            print(f"    {uni}: {count}")

    return df[COURSE_FIELDS]


def generate_artifacts(
    courses: pd.DataFrame,
    out_path: Path,
    verbose: bool = True,
) -> int:
    """Generate unified_v5 artifacts.jsonl combining all sources."""
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    seen_ids = set()

    with out_path.open("w", encoding="utf-8") as f:
        # === Mission artifacts ===
        mission_source = _MISSION_CORPUS if _MISSION_CORPUS.exists() else _MISSION_STATEMENTS
        if mission_source.exists():
            missions = pd.read_csv(mission_source)
            for _, r in missions.iterrows():
                uni = str(r.get("university", "")).strip()
                text = _norm(str(r.get("text", r.get("mission_text", ""))))
                if not text:
                    continue
                hid = _short_hash(f"MISSION|{uni}|{text}")
                aid = f"mission:{uni}:{hid}"
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)
                rec = {
                    "artifact_id": aid,
                    "type": "MISSION",
                    "source_url": f"urn:ads:unified_v5:mission:{uni}",
                    "source_hash": _sha256_text(text),
                    "timestamp": now,
                    "text": text,
                    "metadata": {
                        "institution": uni,
                        "university": uni,
                        "dataset": "unified_v5",
                        "provenance": {
                            "source_url": f"urn:ads:unified_v5:mission:{uni}",
                            "fetched_at": now,
                            "raw_text_hash": _sha256_text(text),
                            "dataset": "unified_v5",
                        },
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1

            if verbose:
                print(f"  Missions: {count}")

        # === Course artifacts ===
        course_count = 0
        for _, r in courses.iterrows():
            uni = str(r.get("university", "")).strip()
            dept = _norm(r.get("department", ""))
            course_id = str(r.get("course_id", "")).strip()
            name = _norm(r.get("course_name", ""))
            desc = _norm(r.get("description", ""))
            units = r.get("units", None)
            level = _norm(r.get("level", ""))
            prereq = _norm(r.get("prerequisites", ""))
            if not name and not desc:
                continue

            key = f"{uni}|{dept}|{course_id}|{name}"
            hid = _short_hash(key)
            aid = f"course:{uni}:{hid}" if uni else f"course:{hid}"
            if aid in seen_ids:
                continue
            seen_ids.add(aid)

            parts = [
                f"{name}." if name else "",
                f"University: {uni}." if uni else "",
                f"Department: {dept}." if dept else "",
                f"Level: {level}." if level else "",
                f"Units: {units}." if pd.notna(units) else "",
                f"Prerequisites: {prereq}."
                if prereq and str(prereq).lower() != "nan" else "",
                f"Description: {desc}" if desc else "",
            ]
            text = _norm(" ".join(p for p in parts if p))
            prov = {
                "source_url": f"urn:ads:unified_v5:course:{uni}:{hid}",
                "fetched_at": now,
                "raw_text_hash": _sha256_text(text),
                "dataset": "unified_v5",
                "university": uni,
            }
            rec = {
                "artifact_id": aid,
                "type": "COURSE",
                "source_url": prov["source_url"],
                "source_hash": prov["raw_text_hash"],
                "timestamp": now,
                "text": text,
                "metadata": {
                    "institution": uni,
                    "university": uni,
                    "department": dept,
                    "course_id": course_id,
                    "course_name": name,
                    "units": units if pd.notna(units) else None,
                    "level": level,
                    "prerequisites": prereq
                    if prereq and str(prereq).lower() != "nan" else None,
                    "provenance": prov,
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
            course_count += 1

        if verbose:
            print(f"  Courses: {course_count}")

        # === O*NET artifacts (from unified_v3) ===
        onet_path = _REPO_ROOT / "data" / "processed" / "unified_v3" / "onet_occupations.csv"
        onet_count = 0
        if onet_path.exists():
            onet = pd.read_csv(onet_path)
            for _, r in onet.iterrows():
                code = str(r.get("onet_soc_code", "")).strip()
                title = _norm(r.get("title", ""))
                desc = _norm(r.get("description", ""))
                if not title:
                    continue
                text = f"{title}. {desc}" if desc else f"{title}."
                hid = _short_hash(f"onet|{code}")
                aid = f"onet:{hid}"
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)
                rec = {
                    "artifact_id": aid,
                    "type": "JOB_ROLE",
                    "source_url": f"https://www.onetonline.org/link/summary/{code}",
                    "source_hash": _sha256_text(text),
                    "timestamp": now,
                    "text": text,
                    "metadata": {
                        "institution": "MARKET",
                        "onet_soc_code": code,
                        "title": title,
                        "country": "US",
                        "provenance": {
                            "source_url": f"https://www.onetonline.org/link/summary/{code}",
                            "fetched_at": now,
                            "raw_text_hash": _sha256_text(text),
                            "dataset": "unified_v5",
                        },
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
                onet_count += 1

        if verbose:
            print(f"  O*NET: {onet_count}")

        # === Market artifacts (Profstandart, HeadHunter, ESCO) ===
        for label, src_path, _prefix in _MARKET_ARTIFACT_SOURCES:
            if _prefix == "onet:":
                continue  # Already handled above
            if not src_path.exists():
                if verbose:
                    print(f"  [SKIP] {label}: {src_path} not found")
                continue
            market_count = 0
            with src_path.open("r", encoding="utf-8") as src_f:
                for line in src_f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    aid = rec.get("artifact_id", "")
                    if aid in seen_ids:
                        continue
                    seen_ids.add(aid)
                    # Update provenance to unified_v5
                    if "metadata" in rec and "provenance" in rec["metadata"]:
                        rec["metadata"]["provenance"]["dataset"] = "unified_v5"
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
                    market_count += 1
            if verbose:
                print(f"  {label}: {market_count}")

    return count


def main():
    parser = argparse.ArgumentParser(description="Build unified_v5 dataset")
    parser.add_argument("--skip-artifacts", action="store_true")
    args = parser.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Building unified_v5 ===\n")

    # 1. Merge courses
    print("Merging courses...")
    courses = merge_courses()

    # Save courses.csv
    courses_path = _OUT_DIR / "courses.csv"
    courses.to_csv(courses_path, index=False)
    print(f"\n  Saved: {courses_path}")

    # 2. Copy supporting files
    for fname in [
        "mission_corpus.csv", "mission_statements.csv",
        "onet_occupations.csv", "onet_skills.csv", "onet_tasks.csv",
        "onet_technology.csv", "cip_to_soc.csv", "soc_to_cip.csv",
        "programs.csv", "synthetic_outcomes_aggregated.csv",
        "synthetic_outcomes_detailed.csv",
    ]:
        src = _REPO_ROOT / "data" / "processed" / "unified_v3" / fname
        if src.exists():
            shutil.copy2(src, _OUT_DIR / fname)

    # Copy real_outcomes directory
    real_outcomes_src = _REPO_ROOT / "data" / "processed" / "unified_v3" / "real_outcomes"
    real_outcomes_dst = _OUT_DIR / "real_outcomes"
    if real_outcomes_src.exists() and not real_outcomes_dst.exists():
        shutil.copytree(real_outcomes_src, real_outcomes_dst)

    # 3. Generate artifacts
    if not args.skip_artifacts:
        print("\nGenerating artifacts.jsonl...")
        artifacts_path = _OUT_DIR / "artifacts.jsonl"
        total = generate_artifacts(courses, artifacts_path)
        print(f"\n  Total artifacts: {total}")
    else:
        print("\n  [SKIP] Artifact generation")

    # 4. Manifest
    manifest = {
        "dataset": "unified_v5",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "courses": len(courses),
        "universities": sorted(courses["university"].unique().tolist()),
        "labor_markets": ["O*NET (US)", "Profstandart (RU)", "HeadHunter (RU)", "ESCO (EU)"],
        "content_hash": hashlib.sha256(
            courses_path.read_bytes()
        ).hexdigest(),
    }
    manifest_path = _OUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n  Manifest: {manifest_path}")
    print("\n=== unified_v5 complete ===")


if __name__ == "__main__":
    main()
