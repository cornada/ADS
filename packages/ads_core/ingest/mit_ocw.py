"""MIT OpenCourseWare enhanced connector with manifest support.

Enhanced version of OCW connector that:
1. Reads from manifest YAML for course selection
2. Supports mission text ingestion
3. Handles both snapshot and live modes
4. Tracks provenance and license info

License: CC BY-NC-SA 4.0 (MIT OCW standard license)
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

import yaml

from ads_core.data.schemas import Artifact, ArtifactType
from ads_core.ingest.base import BaseConnector, create_provenance, normalize_text


OCW_BASE_URL = "https://ocw.mit.edu"
OCW_LICENSE = "CC BY-NC-SA 4.0"


class SimpleHTMLTextExtractor(HTMLParser):
    """Extract text content from HTML, stripping tags."""

    def __init__(self):
        super().__init__()
        self.text_parts: List[str] = []
        self._skip_tags = {"script", "style", "nav", "footer", "header"}
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self.text_parts.append(text)

    def get_text(self) -> str:
        return " ".join(self.text_parts)


def extract_html_text(html: str) -> str:
    """Extract plain text from HTML content."""
    parser = SimpleHTMLTextExtractor()
    parser.feed(html)
    return normalize_text(parser.get_text())


def extract_ocw_course_description(html: str) -> Dict[str, str]:
    """Extract course description and metadata from OCW course page."""
    result = {"description": "", "objectives": "", "topics": ""}

    # Extract description from meta tags or main content
    desc_match = re.search(r'<meta name="description" content="([^"]+)"', html)
    if desc_match:
        result["description"] = desc_match.group(1)

    # Look for course description section
    desc_section = re.search(
        r'(?:Course Description|About this Course)[:\s]*</[^>]+>\s*<[^>]+>([^<]+)',
        html,
        re.IGNORECASE
    )
    if desc_section:
        result["description"] = desc_section.group(1).strip()

    # Extract learning objectives if present
    obj_match = re.search(
        r'(?:Learning Objectives|Course Goals)[:\s]*</[^>]+>\s*<[^>]+>([^<]+)',
        html,
        re.IGNORECASE
    )
    if obj_match:
        result["objectives"] = obj_match.group(1).strip()

    return result


class MitOcwConnector(BaseConnector):
    """Enhanced OCW connector with manifest support.

    Features:
    - Manifest-driven course selection
    - Mission text ingestion
    - Provenance and license tracking
    - Snapshot and live modes
    """

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        snapshot_dir: Optional[Path] = None,
        max_courses: Optional[int] = None,
        live_fetch: bool = False,
        include_missions: bool = True,
    ):
        """Initialize enhanced OCW connector.

        Args:
            manifest_path: Path to mit_sources.yaml manifest
            snapshot_dir: Directory with pre-downloaded course data
            max_courses: Limit courses (for testing)
            live_fetch: Enable fetching from web
            include_missions: Include mission texts
        """
        self.manifest_path = manifest_path
        self.snapshot_dir = snapshot_dir
        self.max_courses = max_courses
        self.live_fetch = live_fetch
        self.include_missions = include_missions
        self._manifest: Optional[Dict] = None

    @property
    def source_name(self) -> str:
        return "mit_ocw_enhanced"

    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest configuration."""
        if self._manifest is not None:
            return self._manifest

        if self.manifest_path and Path(self.manifest_path).exists():
            self._manifest = yaml.safe_load(
                Path(self.manifest_path).read_text(encoding="utf-8")
            )
        else:
            # Default manifest
            self._manifest = {
                "ocw_courses": [],
                "missions": [],
                "settings": {"max_courses": self.max_courses},
            }

        return self._manifest

    def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch content from URL."""
        if not self.live_fetch:
            return None

        try:
            req = Request(url, headers={"User-Agent": "ADS-Research/1.0"})
            with urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except URLError as e:
            print(f"Warning: Could not fetch {url}: {e}")
            return None

    def _load_course_data(self, course_config: Dict) -> Optional[Dict[str, Any]]:
        """Load course data from snapshot or web."""
        number = course_config.get("number", "")
        url = course_config.get("url", "")

        # Try snapshot first
        if self.snapshot_dir:
            # Convert course number to filename
            safe_name = number.replace(".", "_").replace(" ", "_")
            snapshot_file = Path(self.snapshot_dir) / f"{safe_name}.json"
            if snapshot_file.exists():
                return json.loads(snapshot_file.read_text(encoding="utf-8"))

        # Try fixture
        fixture_dir = Path(__file__).parent.parent.parent.parent / "data" / "fixtures" / "ocw"
        if fixture_dir.exists():
            courses_file = fixture_dir / "courses.json"
            if courses_file.exists():
                courses = json.loads(courses_file.read_text(encoding="utf-8"))
                for course in courses:
                    if course.get("course_number") == number:
                        return course

        # Live fetch
        if url and self.live_fetch:
            html = self._fetch_url(url)
            if html:
                extracted = extract_ocw_course_description(html)
                return {
                    "course_number": number,
                    "title": course_config.get("title", ""),
                    "description": extracted.get("description", ""),
                    "url": url,
                }

        # Return minimal data from manifest
        return {
            "course_number": number,
            "title": course_config.get("title", ""),
            "description": "",
            "url": url,
        }

    def _load_mission_text(self, mission_config: Dict) -> Optional[str]:
        """Load mission text from URL or snapshot."""
        mission_id = mission_config.get("id", "")
        url = mission_config.get("url", "")

        # Try snapshot
        if self.snapshot_dir:
            snapshot_file = Path(self.snapshot_dir) / f"mission_{mission_id}.txt"
            if snapshot_file.exists():
                return snapshot_file.read_text(encoding="utf-8")

        # Try fixture
        fixture_dir = Path(__file__).parent.parent.parent.parent / "data" / "fixtures" / "mit_missions"
        fixture_file = fixture_dir / f"{mission_id}.txt"
        if fixture_file.exists():
            return fixture_file.read_text(encoding="utf-8")

        # Live fetch
        if url and self.live_fetch:
            html = self._fetch_url(url)
            if html:
                return extract_html_text(html)

        return None

    def _build_course_text(self, course: Dict[str, Any]) -> str:
        """Build rich text description for a course."""
        parts = []

        number = course.get("course_number", "")
        title = course.get("title", "")
        if number and title:
            parts.append(f"{number} {title}:")
        elif title:
            parts.append(f"{title}:")

        desc = course.get("description", "")
        if desc:
            # Clean HTML if present
            desc = re.sub(r"<[^>]+>", " ", desc)
            parts.append(desc)

        topics = course.get("topics", [])
        if topics:
            if isinstance(topics, list):
                topic_strs = [str(t) for t in topics[:10]]
                parts.append(f"Topics: {', '.join(topic_strs)}.")

        objectives = course.get("learning_objectives", "")
        if objectives:
            if isinstance(objectives, list):
                objectives = " ".join(objectives)
            parts.append(f"Objectives: {objectives}")

        return normalize_text(" ".join(parts))

    def fetch(self) -> Iterator[Artifact]:
        """Fetch artifacts from OCW data."""
        manifest = self._load_manifest()
        count = 0

        # Process mission texts first
        if self.include_missions:
            for mission in manifest.get("missions", []):
                mission_id = mission.get("id", "")
                mission_name = mission.get("name", "")
                mission_url = mission.get("url", "")

                text = self._load_mission_text(mission)
                if not text:
                    # Use placeholder for missions
                    text = f"{mission_name}: Mission statement for {mission_id}."

                text = normalize_text(text)

                prov = create_provenance(
                    source_url=mission_url,
                    raw_text=text,
                    snapshot_path=self.snapshot_dir,
                    mission_id=mission_id,
                )

                yield Artifact(
                    artifact_id=f"mission:{mission_id}",
                    type=ArtifactType.MISSION,
                    source_url=mission_url,
                    source_hash=prov.raw_text_hash,
                    timestamp=prov.fetched_at,
                    text=text,
                    metadata={
                        "mission_id": mission_id,
                        "name": mission_name,
                        "provenance": prov.to_dict(),
                    },
                )

        # Process OCW courses
        for course_config in manifest.get("ocw_courses", []):
            if self.max_courses and count >= self.max_courses:
                break

            course_data = self._load_course_data(course_config)
            if not course_data:
                continue

            number = course_data.get("course_number", "")
            title = course_data.get("title", course_config.get("title", ""))
            url = course_config.get("url", "")

            text = self._build_course_text(course_data)
            if not text.strip():
                # Use minimal text from manifest
                text = f"{number} {title}: MIT OpenCourseWare course."

            prov = create_provenance(
                source_url=url,
                raw_text=text,
                snapshot_path=self.snapshot_dir,
                course_number=number,
                license=OCW_LICENSE,
            )

            artifact_id = f"ocw:{number}".replace(" ", "_")

            # Extract department from course number
            dept = number.split(".")[0] if "." in number else ""

            yield Artifact(
                artifact_id=artifact_id,
                type=ArtifactType.COURSE,
                source_url=url,
                source_hash=prov.raw_text_hash,
                timestamp=prov.fetched_at,
                text=text,
                metadata={
                    "course_number": number,
                    "title": title,
                    "department": dept,
                    "license": OCW_LICENSE,
                    "provenance": prov.to_dict(),
                },
            )
            count += 1


def create_mit_fixtures(output_base: Path) -> None:
    """Create all MIT-related fixtures for testing."""
    # OCW fixture (reuse existing if present)
    from ads_core.ingest.ocw import create_ocw_fixture
    ocw_dir = output_base / "ocw"
    if not (ocw_dir / "courses.json").exists():
        create_ocw_fixture(ocw_dir)

    # MIT catalog fixture
    from ads_core.ingest.mit_catalog import create_catalog_fixture
    catalog_dir = output_base / "mit_catalog"
    if not (catalog_dir / "dept_6.html").exists():
        create_catalog_fixture(catalog_dir, dept_id="6")

    # Mission fixtures
    missions_dir = output_base / "mit_missions"
    missions_dir.mkdir(parents=True, exist_ok=True)

    # MIT institutional mission
    mit_mission = """
MIT Mission Statement

The mission of MIT is to advance knowledge and educate students in science, technology,
and other areas of scholarship that will best serve the nation and the world in the 21st century.

The Institute is committed to generating, disseminating, and preserving knowledge,
and to working with others to bring this knowledge to bear on the world's great challenges.

MIT is dedicated to providing its students with an education that combines rigorous
academic study and the excitement of discovery with the support and intellectual
stimulation of a diverse campus community.
"""
    (missions_dir / "mit_mission.txt").write_text(mit_mission.strip(), encoding="utf-8")

    # EECS department mission
    eecs_mission = """
EECS Department Mission

The MIT Department of Electrical Engineering and Computer Science (EECS) offers
programs that educate students in the core theories and practices of electrical
engineering and computer science.

Our mission is to educate leaders who will advance knowledge and develop innovative
technologies that address the challenges facing our world. We aim to create an
inclusive environment where creativity and intellectual risk-taking are encouraged.

EECS research spans the breadth of electrical engineering and computer science,
from circuits and devices to artificial intelligence and machine learning,
from computer systems and networks to signal processing and communications.
"""
    (missions_dir / "eecs_mission.txt").write_text(eecs_mission.strip(), encoding="utf-8")

    # Write metadata
    meta = {
        "source": "MIT fixtures for ADS testing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contents": {
            "ocw": "MIT OpenCourseWare course data",
            "mit_catalog": "MIT Course Catalog HTML",
            "mit_missions": "Mission statement texts",
        },
    }
    (output_base / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
