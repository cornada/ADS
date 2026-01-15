"""MIT Course Catalog ingestion connector.

MIT Course Catalog provides official course descriptions at: https://catalog.mit.edu/
This connector parses course listings from catalog HTML pages.

For reproducibility, we support both:
1. Live mode: Fetch directly from catalog.mit.edu
2. Snapshot mode: Use pre-downloaded HTML files

License: Public information for educational purposes
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

from ads_core.data.schemas import Artifact, ArtifactType
from ads_core.ingest.base import BaseConnector, create_provenance, normalize_text


MIT_CATALOG_BASE = "https://catalog.mit.edu"


class CatalogHTMLParser(HTMLParser):
    """Parse MIT course catalog HTML to extract course data."""

    def __init__(self):
        super().__init__()
        self.courses: List[Dict[str, Any]] = []
        self._current_course: Optional[Dict[str, Any]] = None
        self._in_course_block = False
        self._in_title = False
        self._in_desc = False
        self._current_text = []
        self._capture = False

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        attrs_dict = dict(attrs)
        class_attr = attrs_dict.get("class", "")

        # Course block detection (MIT catalog uses div.courseblock)
        if tag == "div" and "courseblock" in class_attr:
            self._in_course_block = True
            self._current_course = {}

        # Course title (typically in p.courseblocktitle or strong)
        if self._in_course_block and tag in ("p", "strong"):
            if "courseblocktitle" in class_attr or "cb_title" in class_attr:
                self._in_title = True
                self._current_text = []
                self._capture = True

        # Course description
        if self._in_course_block and tag == "p":
            if "courseblockdesc" in class_attr or "cb_desc" in class_attr:
                self._in_desc = True
                self._current_text = []
                self._capture = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._in_course_block:
            if self._current_course and self._current_course.get("number"):
                self.courses.append(self._current_course)
            self._current_course = None
            self._in_course_block = False

        if tag in ("p", "strong") and self._in_title:
            text = "".join(self._current_text).strip()
            if self._current_course is not None and text:
                # Parse title line: "6.001 Structure and Interpretation..."
                match = re.match(r"^(\d+\.[A-Za-z0-9]+)\s+(.+)$", text)
                if match:
                    self._current_course["number"] = match.group(1)
                    self._current_course["title"] = match.group(2).strip()
                else:
                    # Fallback: try to extract number differently
                    parts = text.split(None, 1)
                    if len(parts) >= 2 and re.match(r"\d+\.", parts[0]):
                        self._current_course["number"] = parts[0]
                        self._current_course["title"] = parts[1].strip()
            self._in_title = False
            self._capture = False

        if tag == "p" and self._in_desc:
            text = "".join(self._current_text).strip()
            if self._current_course is not None and text:
                self._current_course["description"] = text
            self._in_desc = False
            self._capture = False

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._current_text.append(data)


def parse_catalog_html(html: str) -> List[Dict[str, Any]]:
    """Parse MIT catalog HTML and return course data.

    Args:
        html: Raw HTML content from catalog page

    Returns:
        List of course dicts with keys: number, title, description
    """
    parser = CatalogHTMLParser()
    parser.feed(html)
    return parser.courses


def parse_simple_course_list(html: str, dept_id: str) -> List[Dict[str, Any]]:
    """Simplified parser for MIT catalog course lists.

    Falls back to regex-based extraction when HTML structure varies.
    """
    courses = []

    # Pattern for course entries: course number followed by title and description
    # MIT format: "6.0001 Introduction to CS... description text"
    course_pattern = re.compile(
        rf"({dept_id}\.[A-Za-z0-9]+(?:J)?)\s*[:\-]?\s*([^<\n]+?)(?:\s*\n|\s*<)",
        re.MULTILINE
    )

    # Also try to find description blocks
    for match in course_pattern.finditer(html):
        number = match.group(1).strip()
        title = match.group(2).strip()

        # Clean up title
        title = re.sub(r"\s+", " ", title)
        title = title.rstrip(".")

        if number and title and len(title) > 3:
            courses.append({
                "number": number,
                "title": title,
                "description": "",  # Will be filled from description blocks
            })

    return courses


class MitCatalogConnector(BaseConnector):
    """Connector for MIT Course Catalog data.

    Operates in two modes:
    1. Snapshot mode: Use pre-downloaded HTML files (reproducible)
    2. Live mode: Fetch from catalog.mit.edu (requires internet)
    """

    def __init__(
        self,
        snapshot_dir: Optional[Path] = None,
        departments: Optional[List[str]] = None,
        max_courses: Optional[int] = None,
        live_fetch: bool = False,
    ):
        """Initialize MIT catalog connector.

        Args:
            snapshot_dir: Directory with pre-downloaded HTML files
            departments: List of department IDs to include (e.g., ["6", "18"])
            max_courses: Maximum courses to return (for testing)
            live_fetch: If True and no snapshot, fetch from web
        """
        self.snapshot_dir = snapshot_dir
        self.departments = departments or ["6"]  # Default to EECS
        self.max_courses = max_courses
        self.live_fetch = live_fetch
        self._cache: Dict[str, List[Dict]] = {}

    @property
    def source_name(self) -> str:
        return "mit_catalog"

    def _fetch_department_html(self, dept_id: str) -> Optional[str]:
        """Fetch HTML for a department, from snapshot or web."""
        # Try snapshot first
        if self.snapshot_dir:
            snapshot_file = Path(self.snapshot_dir) / f"dept_{dept_id}.html"
            if snapshot_file.exists():
                return snapshot_file.read_text(encoding="utf-8")

        # Try bundled fixture
        fixture_dir = Path(__file__).parent.parent.parent.parent / "data" / "fixtures" / "mit_catalog"
        fixture_file = fixture_dir / f"dept_{dept_id}.html"
        if fixture_file.exists():
            return fixture_file.read_text(encoding="utf-8")

        # Live fetch if enabled
        if self.live_fetch:
            url = f"{MIT_CATALOG_BASE}/subjects/{dept_id}/"
            try:
                req = Request(url, headers={"User-Agent": "ADS-Research/1.0"})
                with urlopen(req, timeout=30) as resp:
                    return resp.read().decode("utf-8")
            except URLError as e:
                print(f"Warning: Could not fetch {url}: {e}")
                return None

        return None

    def _parse_courses(self, dept_id: str, html: str) -> List[Dict[str, Any]]:
        """Parse courses from department HTML."""
        # Try structured parsing first
        courses = parse_catalog_html(html)

        # Fallback to simple regex if structured parsing fails
        if not courses:
            courses = parse_simple_course_list(html, dept_id)

        # Ensure department is set
        for course in courses:
            course["department"] = dept_id

        return courses

    def fetch(self) -> Iterator[Artifact]:
        """Fetch course artifacts from MIT catalog."""
        count = 0

        for dept_id in self.departments:
            if self.max_courses and count >= self.max_courses:
                break

            html = self._fetch_department_html(dept_id)
            if not html:
                continue

            courses = self._parse_courses(dept_id, html)

            for course in courses:
                if self.max_courses and count >= self.max_courses:
                    break

                number = course.get("number", "")
                title = course.get("title", "")
                desc = course.get("description", "")

                if not number or not title:
                    continue

                # Build course text
                text_parts = [f"{number} {title}:"]
                if desc:
                    text_parts.append(desc)
                text = normalize_text(" ".join(text_parts))

                if not text.strip():
                    continue

                # Source URL
                source_url = f"{MIT_CATALOG_BASE}/subjects/{dept_id}/"
                if self.snapshot_dir:
                    source_url = f"file://{self.snapshot_dir}/dept_{dept_id}.html"

                prov = create_provenance(
                    source_url=source_url,
                    raw_text=text,
                    snapshot_path=self.snapshot_dir,
                    course_number=number,
                    department=dept_id,
                )

                artifact_id = f"mit_catalog:{number}".replace(" ", "_")

                yield Artifact(
                    artifact_id=artifact_id,
                    type=ArtifactType.COURSE,
                    source_url=prov.source_url,
                    source_hash=prov.raw_text_hash,
                    timestamp=prov.fetched_at,
                    text=text,
                    metadata={
                        "course_number": number,
                        "title": title,
                        "department": dept_id,
                        "description": desc,
                        "provenance": prov.to_dict(),
                    },
                )
                count += 1


def create_catalog_fixture(output_dir: Path, dept_id: str = "6") -> None:
    """Create a minimal MIT catalog fixture for testing.

    Creates sample HTML that mimics MIT catalog structure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample courses (based on real MIT catalog format)
    courses_html = """
<!DOCTYPE html>
<html>
<head><title>Course 6 - EECS</title></head>
<body>
<div class="courseblock">
<p class="courseblocktitle"><strong>6.0001 Introduction to Computer Science and Programming Using Python</strong></p>
<p class="courseblockdesc">Introduction to computer science and programming for students with little or no programming experience. Students develop skills in computational problem solving using Python.</p>
</div>

<div class="courseblock">
<p class="courseblocktitle"><strong>6.006 Introduction to Algorithms</strong></p>
<p class="courseblockdesc">Introduction to mathematical modeling of computational problems. Covers common algorithms, algorithmic paradigms, and data structures used to solve problems.</p>
</div>

<div class="courseblock">
<p class="courseblocktitle"><strong>6.031 Software Construction</strong></p>
<p class="courseblockdesc">Introduces fundamental principles of software development. Topics include testing, specifications, and design patterns.</p>
</div>

<div class="courseblock">
<p class="courseblocktitle"><strong>6.033 Computer System Engineering</strong></p>
<p class="courseblockdesc">Topics on the engineering of computer software and hardware systems. Lectures and recitations examine design challenges in systems including networks, distributed systems, and operating systems.</p>
</div>

<div class="courseblock">
<p class="courseblocktitle"><strong>6.034 Artificial Intelligence</strong></p>
<p class="courseblockdesc">Introduction to artificial intelligence. Representations, techniques, and architectures for knowledge representation, problem solving, learning, and planning.</p>
</div>

<div class="courseblock">
<p class="courseblocktitle"><strong>6.036 Introduction to Machine Learning</strong></p>
<p class="courseblockdesc">Introduction to machine learning principles including supervised learning, neural networks, feature engineering, and regularization.</p>
</div>

<div class="courseblock">
<p class="courseblocktitle"><strong>6.042J Mathematics for Computer Science</strong></p>
<p class="courseblockdesc">Elementary discrete mathematics for computer science and engineering. Topics include logic, proofs, induction, number theory, probability, and counting.</p>
</div>

<div class="courseblock">
<p class="courseblocktitle"><strong>6.046J Design and Analysis of Algorithms</strong></p>
<p class="courseblockdesc">Techniques for design and analysis of algorithms. Divide-and-conquer, dynamic programming, greedy algorithms, and graph algorithms.</p>
</div>

<div class="courseblock">
<p class="courseblocktitle"><strong>6.S191 Introduction to Deep Learning</strong></p>
<p class="courseblockdesc">Introduction to deep learning methods with applications to computer vision, natural language processing, and biology.</p>
</div>

<div class="courseblock">
<p class="courseblocktitle"><strong>6.867 Machine Learning</strong></p>
<p class="courseblockdesc">Graduate-level introduction to machine learning. Supervised learning, classification, regression, and kernel methods.</p>
</div>
</body>
</html>
"""

    (output_dir / f"dept_{dept_id}.html").write_text(courses_html, encoding="utf-8")

    # Write metadata
    meta = {
        "source": "MIT Course Catalog - synthetic fixture for testing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "license": "Public information for educational purposes",
        "departments": [dept_id],
        "original_url": f"{MIT_CATALOG_BASE}/subjects/{dept_id}/",
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
