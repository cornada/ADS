"""Fetch multimodal metadata from MIT OCW for ADS dataset augmentation.

For each MIT OCW course, fetches data.json and extracts:
- learning_resource_types (Lecture Videos, Lecture Notes, etc.)
- topics (hierarchical subject classification)
- level, year, term
- image_src (course cover image URL)
- course_image_metadata

Outputs: data/multimodal/ocw_video_metadata.jsonl

Usage:
    python scripts/fetch_ocw_multimodal.py [--max-courses N] [--delay SECONDS]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
SITEMAP_URL = "https://ocw.mit.edu/sitemap.xml"
OCW_BASE = "https://ocw.mit.edu"
USER_AGENT = "ADS-Research/2.0 (academic; multimodal-augmentation)"

# Multimodal resource categories (based on actual OCW resource type values)
VIDEO_TYPES = {
    "Lecture Videos", "Tutorial Videos", "Demonstration Videos",
    "Simulation Videos", "Problem-solving Videos", "Competition Videos",
    "Other Video", "Videos", "Video Materials",
}
AUDIO_TYPES = {"Lecture Audio", "Podcasts", "Demonstration Audio"}
NOTES_TYPES = {"Lecture Notes", "Readings", "Open Textbooks"}
VISUAL_TYPES = {"Image Gallery", "Instructor Insights"}
INTERACTIVE_TYPES = {"Simulations", "Editable Files"}
ASSESSMENT_TYPES = {
    "Problem Sets", "Exams", "Problem Set Solutions", "Exam Solutions",
    "Supplemental Exam Materials",
}


def _fetch(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch URL content with polite headers."""
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None


def fetch_sitemap_course_slugs() -> List[str]:
    """Parse MIT OCW sitemap.xml and extract all course slugs."""
    logger.info("Fetching sitemap from %s", SITEMAP_URL)
    xml_text = _fetch(SITEMAP_URL, timeout=60)
    if not xml_text:
        raise RuntimeError("Could not fetch sitemap.xml")

    root = ET.fromstring(xml_text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    slugs = []
    for sitemap in root.findall("sm:sitemap", ns):
        loc = sitemap.find("sm:loc", ns)
        if loc is not None and loc.text:
            # Extract slug from URL like:
            # https://ocw.mit.edu/courses/6-006-intro-to-algorithms-spring-2020/sitemap.xml
            match = re.search(r"/courses/([^/]+)/sitemap\.xml", loc.text)
            if match:
                slugs.append(match.group(1))

    logger.info("Found %d course slugs in sitemap", len(slugs))
    return slugs


def slug_to_course_number(slug: str) -> Optional[str]:
    """Extract MIT catalog course number from OCW slug.

    Examples:
        '6-006-introduction-to-algorithms-spring-2020' -> '6.006'
        '18-06-linear-algebra-spring-2010' -> '18.06'
        '6-0001-intro-cs-programming-python-fall-2016' -> '6.0001'
        '15-093j-optimization-methods-fall-2009' -> '15.093J'
        '11-s196-global-freshwater-spring-2011' -> '11.S196'
        '16-a47-the-engineer-of-2020-fall-2009' -> '16.A47'
        '15-es718-global-health-spring-2015' -> '15.ES718'
    """
    # Pattern: dept-number_or_letter_prefix (e.g., "6-006", "11-s196", "16-a47", "15-es718")
    match = re.match(r"^(\d+)-([a-zA-Z]*\d+[a-zA-Z]?)", slug)
    if not match:
        return None

    dept = match.group(1)
    num_part = match.group(2)

    # Reconstruct: "6" + "006" -> "6.006", "11" + "s196" -> "11.S196"
    course_num = f"{dept}.{num_part.upper()}"

    return course_num


def fetch_course_data(slug: str) -> Optional[Dict[str, Any]]:
    """Fetch data.json for a specific OCW course."""
    url = f"{OCW_BASE}/courses/{slug}/data.json"
    text = _fetch(url)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON from %s", url)
        return None


def extract_multimodal_features(data: Dict[str, Any], slug: str) -> Dict[str, Any]:
    """Extract multimodal metadata features from OCW data.json."""
    resource_types = set(data.get("learning_resource_types", []))

    # Binary features
    has_video = bool(resource_types & VIDEO_TYPES)
    has_audio = bool(resource_types & AUDIO_TYPES)
    has_notes = bool(resource_types & NOTES_TYPES)
    has_visual = bool(resource_types & VISUAL_TYPES)
    has_interactive = bool(resource_types & INTERACTIVE_TYPES)
    has_assessment = bool(resource_types & ASSESSMENT_TYPES)

    # Multimedia flag: any non-text media (video, audio, visual, interactive)
    has_multimedia = has_video or has_audio or has_visual or has_interactive

    # Resource richness: count of distinct resource type categories (expanded to 6)
    richness = sum([has_video, has_audio, has_notes, has_visual, has_interactive, has_assessment])

    # Image availability
    image_src = data.get("image_src", "")
    has_image = bool(image_src)

    # Topic hierarchy
    topics = data.get("topics", [])
    flat_topics = []
    for topic_path in topics:
        if isinstance(topic_path, list):
            flat_topics.extend(topic_path)
        elif isinstance(topic_path, str):
            flat_topics.append(topic_path)

    # Course metadata
    level_raw = data.get("level", [])
    if isinstance(level_raw, list):
        level = level_raw[0] if level_raw else ""
    else:
        level = str(level_raw)

    return {
        "ocw_slug": slug,
        "ocw_url": f"{OCW_BASE}/courses/{slug}/",
        "course_title": data.get("course_title", ""),
        "primary_course_number": data.get("primary_course_number", ""),
        "department_numbers": data.get("department_numbers", []),
        "term": data.get("term", ""),
        "year": data.get("year", ""),
        "level": level,
        # Multimodal features
        "learning_resource_types": sorted(resource_types),
        "has_video": has_video,
        "has_audio": has_audio,
        "has_notes": has_notes,
        "has_visual": has_visual,
        "has_interactive": has_interactive,
        "has_assessment": has_assessment,
        "has_multimedia": has_multimedia,
        "resource_richness": richness,
        "resource_type_count": len(resource_types),
        # Visual
        "has_image": has_image,
        "image_src": image_src,
        # Topics
        "topics": topics,
        "topic_count": len(flat_topics),
        # Instructors count (proxy for course scale)
        "instructor_count": len(data.get("instructors", [])),
    }


def load_ads_mit_course_ids(artifacts_path: Path) -> Dict[str, str]:
    """Load MIT course IDs from ADS unified dataset.

    Returns: {course_number_upper: artifact_id}
    """
    mapping = {}
    with open(artifacts_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if ":MIT:" in obj.get("artifact_id", ""):
                meta = obj.get("metadata", {})
                cid = meta.get("course_id", "")
                if cid:
                    key = cid.upper().rstrip(".")
                    mapping[key] = obj["artifact_id"]
    return mapping


def run(
    max_courses: Optional[int] = None,
    delay: float = 0.5,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the multimodal metadata collection pipeline."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "multimodal"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get all OCW course slugs from sitemap
    slugs = fetch_sitemap_course_slugs()

    # Step 2: Build slug -> course_number mapping
    slug_to_num: Dict[str, str] = {}
    for slug in slugs:
        num = slug_to_course_number(slug)
        if num:
            slug_to_num[slug] = num

    logger.info("Mapped %d/%d slugs to course numbers", len(slug_to_num), len(slugs))

    # Step 3: Load ADS MIT course IDs for matching
    artifacts_path = PROJECT_ROOT / "data" / "processed" / "unified_v4" / "artifacts.jsonl"
    if artifacts_path.exists():
        ads_courses = load_ads_mit_course_ids(artifacts_path)
        logger.info("Loaded %d MIT courses from ADS dataset", len(ads_courses))
    else:
        ads_courses = {}
        logger.warning("ADS unified_v4 not found at %s", artifacts_path)

    # Step 4: Fetch data.json for each course
    results = []
    errors = []
    matched = 0

    process_slugs = list(slug_to_num.keys())
    if max_courses:
        process_slugs = process_slugs[:max_courses]

    for i, slug in enumerate(process_slugs):
        course_num = slug_to_num[slug]

        if (i + 1) % 50 == 0:
            logger.info("Progress: %d/%d courses fetched (%d matched ADS)",
                        i + 1, len(process_slugs), matched)

        data = fetch_course_data(slug)
        if data is None:
            errors.append({"slug": slug, "error": "fetch_failed"})
            time.sleep(delay)
            continue

        features = extract_multimodal_features(data, slug)

        # Check ADS match
        num_upper = course_num.upper()
        ads_match = ads_courses.get(num_upper, "")
        features["ads_artifact_id"] = ads_match
        features["ads_matched"] = bool(ads_match)
        features["catalog_course_number"] = course_num

        if ads_match:
            matched += 1

        results.append(features)
        time.sleep(delay)

    # Step 5: Write outputs
    output_path = output_dir / "ocw_video_metadata.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    errors_path = output_dir / "ocw_fetch_errors.jsonl"
    with open(errors_path, "w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Step 6: Compute summary statistics
    total = len(results)
    with_video = sum(1 for r in results if r["has_video"])
    with_notes = sum(1 for r in results if r["has_notes"])
    with_image = sum(1 for r in results if r["has_image"])
    with_interactive = sum(1 for r in results if r["has_interactive"])
    with_assessment = sum(1 for r in results if r["has_assessment"])
    mean_richness = sum(r["resource_richness"] for r in results) / max(total, 1)
    mean_resource_types = sum(r["resource_type_count"] for r in results) / max(total, 1)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_ocw_slugs": len(slugs),
        "total_fetched": total,
        "fetch_errors": len(errors),
        "ads_matched": matched,
        "ads_match_rate": f"{matched/max(total,1)*100:.1f}%",
        "multimodal_stats": {
            "has_video": with_video,
            "has_video_pct": f"{with_video/max(total,1)*100:.1f}%",
            "has_notes": with_notes,
            "has_notes_pct": f"{with_notes/max(total,1)*100:.1f}%",
            "has_image": with_image,
            "has_image_pct": f"{with_image/max(total,1)*100:.1f}%",
            "has_interactive": with_interactive,
            "has_assessment": with_assessment,
            "mean_richness": round(mean_richness, 2),
            "mean_resource_type_count": round(mean_resource_types, 2),
        },
    }

    summary_path = output_dir / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("COLLECTION COMPLETE")
    logger.info("  Total fetched: %d", total)
    logger.info("  Errors: %d", len(errors))
    logger.info("  ADS matched: %d (%.1f%%)", matched, matched / max(total, 1) * 100)
    logger.info("  With video: %d (%.1f%%)", with_video, with_video / max(total, 1) * 100)
    logger.info("  With notes: %d (%.1f%%)", with_notes, with_notes / max(total, 1) * 100)
    logger.info("  Mean richness: %.2f", mean_richness)
    logger.info("  Output: %s", output_path)
    logger.info("  Summary: %s", summary_path)
    logger.info("=" * 60)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MIT OCW multimodal metadata")
    parser.add_argument("--max-courses", type=int, default=None,
                        help="Limit number of courses to fetch (for testing)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between requests in seconds (default: 0.5)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/multimodal/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    run(max_courses=args.max_courses, delay=args.delay, output_dir=output_dir)


if __name__ == "__main__":
    main()
