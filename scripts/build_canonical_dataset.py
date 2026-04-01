"""Build canonical unified dataset with stable IDs across all data layers.

Resolves 9 data islands into ONE dataset with:
- Stable canonical_id per course (institution:catalog_number)
- Text layer (course descriptions)
- Multimodal layer (OCW resource types, images, transcripts)
- Instructional activity layer (MISIS hours, labs, lectures)
- Competency layer (FGOS codes, O*NET skills)
- Structural layer (prerequisites)

Output: data/canonical/ads_mm_canonical.jsonl
        data/canonical/join_manifest.json

Usage:
    python scripts/build_canonical_dataset.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# CANONICAL ID EXTRACTION
# =============================================================================

def normalize_catalog_id(raw_id: str, institution: str = "") -> str:
    """Normalize a catalog ID: strip prefixes, collapse whitespace.

    For MISIS, keep the hash suffix (direction_code is program-level, not course-level).
    For others, strip hash suffixes.
    """
    # Strip institution prefixes like "kth:", "edinburgh:"
    for prefix in ["kth:", "edinburgh:", "mit:", "ucb:", "stanford:", "uiuc:", "cornell:"]:
        if raw_id.lower().startswith(prefix):
            raw_id = raw_id[len(prefix):]
            break
    # Strip hash suffixes ONLY for non-MISIS (MISIS needs hash to distinguish courses within a program)
    if institution.upper() != "MISIS" and re.match(r".+:[0-9a-f]{8,}$", raw_id):
        raw_id = raw_id.rsplit(":", 1)[0]
    return raw_id.strip()


def extract_canonical_id(artifact: Dict[str, Any]) -> Optional[str]:
    """Extract stable canonical ID from any artifact record.

    Returns: '{INSTITUTION}:{catalog_id}' or None if not a course.
    """
    aid = artifact.get("artifact_id", "")
    atype = artifact.get("type", "")
    meta = artifact.get("metadata", {})
    institution = meta.get("institution", "")

    # Non-course artifacts
    if atype not in ("COURSE",):
        if atype == "JOB_ROLE":
            return f"ONET:{meta.get('occupation_id', aid)}"
        if atype == "MISSION":
            return f"MISSION:{institution}:{meta.get('mission_id', aid)}"
        if atype == "COMPETENCY":
            return f"COMP:{meta.get('code', aid)}"
        return None

    # Course: extract catalog ID from metadata
    course_id = meta.get("course_id", "")

    if not institution:
        # Try to infer from artifact_id
        parts = aid.split(":")
        if len(parts) >= 2:
            institution = parts[1]

    if not institution:
        return None

    inst = institution.upper().replace("UC BERKELEY", "UCB").replace(" ", "_")

    if course_id:
        canon = normalize_catalog_id(course_id, institution=institution)
        return f"{inst}:{canon}"

    # Fallback: use title hash (last resort)
    title = artifact.get("title", "") or meta.get("course_name", "")
    if title:
        short_hash = hashlib.sha256(title.encode()).hexdigest()[:8]
        return f"{inst}:~{short_hash}"

    return None


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_unified(version: str) -> Dict[str, Dict]:
    """Load unified dataset, return {canonical_id: record}."""
    path = PROJECT_ROOT / "data" / "processed" / version / "artifacts.jsonl"
    if not path.exists():
        logger.warning("Not found: %s", path)
        return {}

    result = {}
    skipped = 0
    for rec in load_jsonl(path):
        cid = extract_canonical_id(rec)
        if cid:
            result[cid] = {
                "canonical_id": cid,
                "artifact_id_original": rec.get("artifact_id", ""),
                "type": rec.get("type", ""),
                "institution": rec.get("metadata", {}).get("institution", ""),
                "title": rec.get("title", "") or rec.get("metadata", {}).get("course_name", ""),
                "text": rec.get("text", ""),
                "source_url": rec.get("source_url", ""),
                "source_hash": rec.get("source_hash", ""),
                "metadata": rec.get("metadata", {}),
                # Layers (filled later)
                "multimodal": {},
                "activity": {},
                "competencies": [],
                "prerequisites": [],
                "transcripts": {},
                "image": {},
            }
        else:
            skipped += 1

    logger.info("Loaded %s: %d records (%d skipped)", version, len(result), skipped)
    return result


def enrich_with_multimodal(
    dataset: Dict[str, Dict],
    mm_path: Path,
    v4_path: Optional[Path] = None,
) -> Tuple[int, int]:
    """Add OCW multimodal metadata via artifact_id join."""
    if not mm_path.exists():
        return 0, 0

    # Build v4 artifact_id → canonical_id mapping
    v4_to_canon: Dict[str, str] = {}
    if v4_path and v4_path.exists():
        for rec in load_jsonl(v4_path):
            cid = extract_canonical_id(rec)
            if cid:
                v4_to_canon[rec.get("artifact_id", "")] = cid

    matched = 0
    total = 0
    for rec in load_jsonl(mm_path):
        total += 1
        v4_aid = rec.get("ads_artifact_id", "")
        cid = v4_to_canon.get(v4_aid)

        if not cid:
            # Try direct catalog number match
            cat_num = rec.get("catalog_course_number", "")
            if cat_num:
                cid = f"MIT:{cat_num}"

        if cid and cid in dataset:
            dataset[cid]["multimodal"] = {
                "ocw_slug": rec.get("ocw_slug", ""),
                "has_video": rec.get("has_video", False),
                "has_notes": rec.get("has_notes", False),
                "has_assessment": rec.get("has_assessment", False),
                "has_interactive": rec.get("has_interactive", False),
                "resource_richness": rec.get("resource_richness", 0),
                "resource_type_count": rec.get("resource_type_count", 0),
                "learning_resource_types": rec.get("learning_resource_types", []),
                "has_image": rec.get("has_image", False),
                "image_src": rec.get("image_src", ""),
                "topic_count": rec.get("topic_count", 0),
            }
            matched += 1

    logger.info("Multimodal enrichment: %d/%d matched", matched, total)
    return matched, total


def enrich_with_misis(dataset: Dict[str, Dict], misis_path: Path) -> int:
    """Add MISIS instructional activity + competency data via title join."""
    if not misis_path.exists():
        return 0

    # Build title → misis_record mapping
    misis_by_title: Dict[str, Dict] = {}
    for rec in load_jsonl(misis_path):
        title = rec.get("title", "").strip()
        if title:
            misis_by_title[title.lower()] = rec

    matched = 0
    for cid, entry in dataset.items():
        if "MISIS" not in cid:
            continue

        # Try multiple title sources
        candidates = [
            entry.get("title", "").strip().lower(),
            entry.get("metadata", {}).get("course_name", "").strip().lower(),
        ]
        # Also try text field (first sentence)
        text = entry.get("text", "")
        if text and "." in text:
            candidates.append(text.split(".")[0].strip().lower())

        misis_rec = None
        for title in candidates:
            if not title:
                continue
            misis_rec = misis_by_title.get(title)
            if misis_rec:
                break

        if not misis_rec:
            # Prefix match: first 30 chars
            title = candidates[0] if candidates else ""
            if len(title) > 15:
                for mt, mr in misis_by_title.items():
                    if mt.startswith(title[:30]) or title.startswith(mt[:30]):
                        misis_rec = mr
                        break

        if misis_rec:
            entry["activity"] = {
                "hours": misis_rec.get("hours", {}),
                "has_lectures": misis_rec.get("has_lectures", False),
                "has_labs": misis_rec.get("has_labs", False),
                "has_practical": misis_rec.get("has_practical", False),
                "has_independent": misis_rec.get("has_independent", False),
                "modality_richness": misis_rec.get("modality_richness", 0),
                "credits_zet": misis_rec.get("credits_zet", 0),
            }
            entry["competencies"] = misis_rec.get("competency_codes", [])
            entry["prerequisites"] = misis_rec.get("prerequisites", [])
            matched += 1

    logger.info("MISIS enrichment: %d matched by title", matched)
    return matched


def enrich_with_transcripts(dataset: Dict[str, Dict], transcript_path: Path) -> int:
    """Add OCW transcript data via ocw_slug join."""
    if not transcript_path.exists():
        return 0

    # Build slug → canonical_id mapping from multimodal layer
    slug_to_cid: Dict[str, str] = {}
    for cid, entry in dataset.items():
        slug = entry.get("multimodal", {}).get("ocw_slug", "")
        if slug:
            slug_to_cid[slug] = cid

    matched = 0
    for rec in load_jsonl(transcript_path):
        slug = rec.get("ocw_slug", "")
        cid = slug_to_cid.get(slug)
        if cid and cid in dataset:
            dataset[cid]["transcripts"] = {
                "has_transcript": rec.get("has_transcript", False),
                "lecture_titles": rec.get("lecture_titles", []),
                "lecture_count": len(rec.get("lecture_titles", [])),
                "syllabus_text": rec.get("syllabus_text", "")[:1000],
                "page_text_length": rec.get("page_text_length", 0),
            }
            matched += 1

    logger.info("Transcript enrichment: %d matched by slug", matched)
    return matched


def enrich_with_images(dataset: Dict[str, Dict], manifest_path: Path) -> int:
    """Add image metadata via ocw_slug join."""
    if not manifest_path.exists():
        return 0

    slug_to_cid: Dict[str, str] = {}
    for cid, entry in dataset.items():
        slug = entry.get("multimodal", {}).get("ocw_slug", "")
        if slug:
            slug_to_cid[slug] = cid

    matched = 0
    for rec in load_jsonl(manifest_path):
        slug = rec.get("ocw_slug", "")
        cid = slug_to_cid.get(slug)
        if cid and cid in dataset:
            dataset[cid]["image"] = {
                "path": rec.get("image_path", ""),
                "size_bytes": rec.get("image_size_bytes", 0),
                "available": rec.get("download_success", False),
            }
            matched += 1

    logger.info("Image enrichment: %d matched by slug", matched)
    return matched


# =============================================================================
# MAIN
# =============================================================================

def build_canonical_dataset() -> Dict[str, Any]:
    """Build the unified canonical dataset."""
    output_dir = PROJECT_ROOT / "data" / "canonical"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load base dataset (use v5 as latest, fallback to v4)
    dataset = load_unified("unified_v5")
    if not dataset:
        dataset = load_unified("unified_v4")

    # Step 2: Enrich with multimodal OCW metadata
    mm_matched, mm_total = enrich_with_multimodal(
        dataset,
        PROJECT_ROOT / "data" / "multimodal" / "ocw_video_metadata.jsonl",
        PROJECT_ROOT / "data" / "processed" / "unified_v4" / "artifacts.jsonl",
    )

    # Step 3: Enrich with MISIS instructional activity
    misis_matched = enrich_with_misis(
        dataset,
        PROJECT_ROOT / "data" / "multimodal" / "misis_courses.jsonl",
    )

    # Step 4: Enrich with transcripts
    transcript_matched = enrich_with_transcripts(
        dataset,
        PROJECT_ROOT / "data" / "multimodal" / "ocw_transcripts.jsonl",
    )

    # Step 5: Enrich with image metadata
    image_matched = enrich_with_images(
        dataset,
        PROJECT_ROOT / "data" / "multimodal" / "ocw_image_manifest.jsonl",
    )

    # Step 6: Write canonical dataset
    out_path = output_dir / "ads_mm_canonical.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for cid in sorted(dataset.keys()):
            entry = dataset[cid]
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    # Step 7: Compute and write manifest
    type_counts = Counter(e["type"] for e in dataset.values())
    inst_counts = Counter(e["institution"] for e in dataset.values())
    has_mm = sum(1 for e in dataset.values() if e.get("multimodal", {}).get("ocw_slug"))
    has_activity = sum(1 for e in dataset.values() if e.get("activity", {}).get("hours"))
    has_comps = sum(1 for e in dataset.values() if e.get("competencies"))
    has_prereqs = sum(1 for e in dataset.values() if e.get("prerequisites"))
    has_transcripts = sum(1 for e in dataset.values() if e.get("transcripts", {}).get("has_transcript"))
    has_images = sum(1 for e in dataset.values() if e.get("image", {}).get("available"))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "ADS-MM canonical dataset with stable IDs and unified data layers",
        "total_records": len(dataset),
        "type_counts": dict(type_counts),
        "institution_counts": dict(inst_counts.most_common()),
        "layers": {
            "text": {"count": len(dataset), "description": "Course/job/mission text descriptions"},
            "multimodal": {"count": has_mm, "description": "MIT OCW resource type annotations"},
            "activity": {"count": has_activity, "description": "MISIS instructional hours breakdown"},
            "competencies": {"count": has_comps, "description": "FGOS competency codes"},
            "prerequisites": {"count": has_prereqs, "description": "Course prerequisite links"},
            "transcripts": {"count": has_transcripts, "description": "OCW lecture titles + syllabi"},
            "images": {"count": has_images, "description": "Course cover images"},
        },
        "join_stats": {
            "multimodal_matched": mm_matched,
            "multimodal_total": mm_total,
            "misis_matched": misis_matched,
            "transcript_matched": transcript_matched,
            "image_matched": image_matched,
        },
        "output_path": str(out_path),
    }

    manifest_path = output_dir / "join_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    logger.info("=" * 60)
    logger.info("CANONICAL DATASET BUILT")
    logger.info("  Total records: %d", len(dataset))
    logger.info("  Types: %s", dict(type_counts))
    logger.info("  Institutions: %d", len(inst_counts))
    logger.info("  With multimodal: %d", has_mm)
    logger.info("  With activity: %d", has_activity)
    logger.info("  With competencies: %d", has_comps)
    logger.info("  With prerequisites: %d", has_prereqs)
    logger.info("  With transcripts: %d", has_transcripts)
    logger.info("  With images: %d", has_images)
    logger.info("  Output: %s", out_path)
    logger.info("  Manifest: %s", manifest_path)
    logger.info("=" * 60)

    return manifest


if __name__ == "__main__":
    build_canonical_dataset()
