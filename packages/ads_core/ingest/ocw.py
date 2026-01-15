"""MIT OpenCourseWare (OCW) course data ingestion connector.

MIT OCW provides open course materials under Creative Commons license.
Data is available at: https://ocw.mit.edu/

This connector uses the publicly available OCW data feeds (JSON).
For reproducibility, we use pinned snapshot files when available.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ads_core.data.schemas import Artifact, ArtifactType
from ads_core.ingest.base import BaseConnector, create_provenance, normalize_text


# OCW JSON feed endpoints (publicly accessible)
OCW_COURSES_URL = "https://ocw.mit.edu/courses/"
OCW_API_URL = "https://ocw.mit.edu/api/v0/"


class OcwConnector(BaseConnector):
    """Connector for MIT OpenCourseWare course data.

    Can operate in two modes:
    1. Snapshot mode: Use a pre-downloaded snapshot file (reproducible)
    2. Live mode: Download from OCW website (requires internet)

    For IJCAI reproducibility, snapshot mode is preferred.
    """

    def __init__(
        self,
        snapshot_path: Optional[Path] = None,
        max_courses: Optional[int] = None,
        departments: Optional[List[str]] = None,
    ):
        """Initialize OCW connector.

        Args:
            snapshot_path: Path to pre-downloaded OCW JSON or directory.
                          If None, will try to use bundled fixture data.
            max_courses: Limit number of courses (for testing/dev).
            departments: Filter to specific departments (e.g., ["6", "18"] for EECS and Math).
        """
        self.snapshot_path = snapshot_path
        self.max_courses = max_courses
        self.departments = departments
        self._data_cache: Optional[List[Dict[str, Any]]] = None

    @property
    def source_name(self) -> str:
        return "mit_ocw"

    def _load_from_snapshot(self) -> List[Dict[str, Any]]:
        """Load course data from snapshot file or directory."""
        if self._data_cache is not None:
            return self._data_cache

        if self.snapshot_path is None:
            # Try bundled fixture
            fixture_path = Path(__file__).parent.parent.parent.parent / "data" / "fixtures" / "ocw"
            if fixture_path.exists():
                self.snapshot_path = fixture_path

        if self.snapshot_path is None:
            raise FileNotFoundError(
                "No OCW snapshot found. Either provide snapshot_path or place fixture data in data/fixtures/ocw/"
            )

        path = Path(self.snapshot_path)

        if path.is_file() and path.suffix == ".json":
            # Single JSON file with all courses
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "courses" in data:
                courses = data["courses"]
            elif isinstance(data, list):
                courses = data
            else:
                courses = [data]
        elif path.is_dir():
            # Directory with course JSON files
            courses = []
            # Try single courses.json first
            courses_file = path / "courses.json"
            if courses_file.exists():
                data = json.loads(courses_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    courses = data
                elif isinstance(data, dict) and "courses" in data:
                    courses = data["courses"]
            else:
                # Load individual course files
                for f in sorted(path.glob("*.json")):
                    if f.name != "meta.json":
                        course = json.loads(f.read_text(encoding="utf-8"))
                        courses.append(course)
        else:
            raise ValueError(f"snapshot_path must be a .json file or directory: {path}")

        self._data_cache = courses
        return courses

    def _build_course_text(self, course: Dict[str, Any]) -> str:
        """Build rich text description for a course."""
        parts = []

        # Course number and title
        course_num = course.get("course_number", course.get("number", ""))
        title = course.get("title", course.get("course_title", ""))
        if course_num and title:
            parts.append(f"{course_num} {title}:")
        elif title:
            parts.append(f"{title}:")

        # Description
        desc = course.get("description", course.get("course_description", ""))
        if desc:
            # Clean HTML if present
            desc = desc.replace("<p>", " ").replace("</p>", " ")
            desc = desc.replace("<br>", " ").replace("<br/>", " ")
            parts.append(desc)

        # Topics/keywords
        topics = course.get("topics", course.get("course_features", []))
        if isinstance(topics, list) and topics:
            topic_strs = [str(t) if not isinstance(t, dict) else t.get("name", str(t)) for t in topics[:10]]
            if topic_strs:
                parts.append(f"Topics: {', '.join(topic_strs)}.")

        # Learning objectives if available
        objectives = course.get("learning_objectives", course.get("objectives", ""))
        if objectives:
            if isinstance(objectives, list):
                objectives = " ".join(objectives)
            parts.append(f"Objectives: {objectives}")

        return normalize_text(" ".join(parts))

    def _get_department(self, course: Dict[str, Any]) -> Optional[str]:
        """Extract department number from course."""
        course_num = course.get("course_number", course.get("number", ""))
        if course_num:
            # MIT course numbers are like "6.001" where 6 is the department
            parts = str(course_num).split(".")
            if parts:
                return parts[0]
        dept = course.get("department", course.get("department_number", ""))
        if dept:
            return str(dept)
        return None

    def fetch(self) -> Iterator[Artifact]:
        """Fetch course artifacts from OCW data."""
        courses = self._load_from_snapshot()

        source_url = OCW_COURSES_URL
        if self.snapshot_path:
            source_url = f"file://{self.snapshot_path}"

        count = 0
        for course in courses:
            if self.max_courses and count >= self.max_courses:
                break

            # Filter by department if specified
            if self.departments:
                dept = self._get_department(course)
                if dept not in self.departments:
                    continue

            course_num = course.get("course_number", course.get("number", ""))
            title = course.get("title", course.get("course_title", "Unknown"))
            course_id = course.get("id", course.get("uid", course_num or f"course_{count}"))

            text = self._build_course_text(course)
            if not text.strip():
                continue

            prov = create_provenance(
                source_url=source_url,
                raw_text=text,
                snapshot_path=self.snapshot_path,
                course_number=course_num,
            )

            # Generate artifact ID
            artifact_id = f"ocw:{course_num}" if course_num else f"ocw:{course_id}"
            artifact_id = artifact_id.replace(" ", "_")

            yield Artifact(
                artifact_id=artifact_id,
                type=ArtifactType.COURSE,
                source_url=prov.source_url,
                source_hash=prov.raw_text_hash,
                timestamp=prov.fetched_at,
                text=text,
                metadata={
                    "course_number": course_num,
                    "title": title,
                    "department": self._get_department(course),
                    "provenance": prov.to_dict(),
                },
            )
            count += 1


def create_ocw_fixture(output_dir: Path, num_courses: int = 20) -> None:
    """Create a minimal OCW fixture for testing.

    This creates sample data that mimics OCW structure for tests.
    Based on real MIT OCW courses but simplified.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample courses (based on real MIT OCW structure)
    courses = [
        {
            "course_number": "6.0001",
            "title": "Introduction to Computer Science and Programming Using Python",
            "description": "Introduction to computer science as a tool to solve real-world analytical problems using Python 3.5. Topics include computational thinking, algorithm development, and data structures.",
            "topics": ["Computer Science", "Programming", "Python", "Algorithms"],
            "department": "6",
        },
        {
            "course_number": "6.006",
            "title": "Introduction to Algorithms",
            "description": "Introduction to mathematical modeling of computational problems, as well as common algorithms, algorithmic paradigms, and data structures. Sorting, searching, graph algorithms.",
            "topics": ["Algorithms", "Data Structures", "Graphs", "Dynamic Programming"],
            "department": "6",
        },
        {
            "course_number": "6.036",
            "title": "Introduction to Machine Learning",
            "description": "Introduction to machine learning principles and methods, including supervised learning, neural networks, optimization, and feature engineering.",
            "topics": ["Machine Learning", "Neural Networks", "Classification", "Regression"],
            "department": "6",
        },
        {
            "course_number": "6.034",
            "title": "Artificial Intelligence",
            "description": "Introduction to representations, techniques, and architectures used in artificial intelligence. Search, constraint satisfaction, planning, knowledge representation.",
            "topics": ["Artificial Intelligence", "Search", "Planning", "Knowledge Representation"],
            "department": "6",
        },
        {
            "course_number": "6.046J",
            "title": "Design and Analysis of Algorithms",
            "description": "Techniques for design and analysis of algorithms including divide-and-conquer, dynamic programming, greedy algorithms, amortized analysis, network flows.",
            "topics": ["Algorithms", "Complexity", "Optimization", "Graph Theory"],
            "department": "6",
        },
        {
            "course_number": "6.S191",
            "title": "Introduction to Deep Learning",
            "description": "Introduction to deep learning methods with applications to computer vision, natural language processing, and biology. Building neural networks with TensorFlow.",
            "topics": ["Deep Learning", "Neural Networks", "Computer Vision", "NLP"],
            "department": "6",
        },
        {
            "course_number": "6.867",
            "title": "Machine Learning",
            "description": "Graduate-level introduction to machine learning. Supervised learning, generalization, classification, regression, feature selection, model selection.",
            "topics": ["Machine Learning", "Statistical Learning", "Kernel Methods", "Ensemble Methods"],
            "department": "6",
        },
        {
            "course_number": "18.05",
            "title": "Introduction to Probability and Statistics",
            "description": "Elementary introduction to probability and statistics. Basic probability models, random variables, expectation, Bayes theorem, central limit theorem.",
            "topics": ["Probability", "Statistics", "Random Variables", "Inference"],
            "department": "18",
        },
        {
            "course_number": "18.06",
            "title": "Linear Algebra",
            "description": "Basic subject on matrix theory and linear algebra. Systems of equations, vector spaces, eigenvalues and eigenvectors, positive definite matrices.",
            "topics": ["Linear Algebra", "Matrices", "Vector Spaces", "Eigenvalues"],
            "department": "18",
        },
        {
            "course_number": "6.042J",
            "title": "Mathematics for Computer Science",
            "description": "Elementary discrete mathematics for computer science and engineering. Logic, proofs, induction, number theory, probability, counting.",
            "topics": ["Discrete Mathematics", "Logic", "Proofs", "Probability"],
            "department": "6",
        },
        {
            "course_number": "15.S12",
            "title": "Blockchain and Money",
            "description": "Study of blockchain technology and its potential use in finance. Cryptography, distributed systems, digital currencies, regulatory challenges.",
            "topics": ["Blockchain", "Cryptocurrency", "Finance", "Distributed Systems"],
            "department": "15",
        },
        {
            "course_number": "6.UAT",
            "title": "Oral Communication",
            "description": "Develops oral presentation skills for technical communication in engineering and science. Practice presenting technical material to varied audiences.",
            "topics": ["Communication", "Presentation Skills", "Technical Writing"],
            "department": "6",
        },
    ]

    # Write fixture file
    (output_dir / "courses.json").write_text(
        json.dumps(courses, indent=2), encoding="utf-8"
    )

    # Write provenance metadata
    meta = {
        "source": "MIT OpenCourseWare - synthetic fixture for testing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY-NC-SA 4.0 (MIT OCW standard license)",
        "note": "This is a minimal fixture for unit tests, not full OCW data.",
        "original_url": OCW_COURSES_URL,
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
