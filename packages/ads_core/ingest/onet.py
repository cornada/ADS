"""O*NET occupation data ingestion connector.

O*NET (Occupational Information Network) provides open labor market data.
Data is available at: https://www.onetcenter.org/database.html

This connector uses the publicly available O*NET database files (no API key required).
For reproducibility, we use pinned snapshot files when available.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from zipfile import ZipFile

from ads_core.data.schemas import Artifact, ArtifactType
from ads_core.ingest.base import BaseConnector, create_provenance, normalize_text
from ads_core.utils.hashing import sha256_text

# O*NET database version for reproducibility
ONET_VERSION = "29_1"
ONET_BASE_URL = "https://www.onetcenter.org/dl_files/database"

# Key files we extract from the O*NET database
ONET_FILES = {
    "occupations": "Occupation Data.txt",
    "skills": "Skills.txt",
    "knowledge": "Knowledge.txt",
    "abilities": "Abilities.txt",
    "tasks": "Task Statements.txt",
    "tech_skills": "Technology Skills.txt",
}


class OnetConnector(BaseConnector):
    """Connector for O*NET occupation data.

    Can operate in two modes:
    1. Snapshot mode: Use a pre-downloaded snapshot file (reproducible)
    2. Live mode: Download from O*NET website (requires internet)

    For IJCAI reproducibility, snapshot mode is preferred.
    """

    def __init__(
        self,
        snapshot_path: Optional[Path] = None,
        max_occupations: Optional[int] = None,
        include_skills: bool = True,
        include_tasks: bool = True,
    ):
        """Initialize O*NET connector.

        Args:
            snapshot_path: Path to pre-downloaded O*NET database ZIP or extracted dir.
                          If None, will try to use bundled fixture data.
            max_occupations: Limit number of occupations (for testing/dev).
            include_skills: Include skill requirements in job role text.
            include_tasks: Include task descriptions in job role text.
        """
        self.snapshot_path = snapshot_path
        self.max_occupations = max_occupations
        self.include_skills = include_skills
        self.include_tasks = include_tasks
        self._data_cache: Optional[Dict[str, Any]] = None

    @property
    def source_name(self) -> str:
        return f"onet_{ONET_VERSION}"

    def _load_tsv(self, content: str) -> List[Dict[str, str]]:
        """Parse tab-separated content into list of dicts."""
        reader = csv.DictReader(io.StringIO(content), delimiter="\t")
        return list(reader)

    def _load_from_snapshot(self) -> Dict[str, List[Dict[str, str]]]:
        """Load data from snapshot file or directory."""
        if self._data_cache is not None:
            return self._data_cache

        data: Dict[str, List[Dict[str, str]]] = {}

        if self.snapshot_path is None:
            # Try bundled fixture
            fixture_path = Path(__file__).parent.parent.parent.parent / "data" / "fixtures" / "onet"
            if fixture_path.exists():
                self.snapshot_path = fixture_path

        if self.snapshot_path is None:
            raise FileNotFoundError(
                "No O*NET snapshot found. Either provide snapshot_path or place fixture data in data/fixtures/onet/"
            )

        path = Path(self.snapshot_path)

        if path.suffix == ".zip":
            # Extract from ZIP
            with ZipFile(path, "r") as zf:
                for key, filename in ONET_FILES.items():
                    try:
                        content = zf.read(filename).decode("utf-8")
                        data[key] = self._load_tsv(content)
                    except KeyError:
                        data[key] = []
        elif path.is_dir():
            # Load from directory
            for key, filename in ONET_FILES.items():
                filepath = path / filename
                if filepath.exists():
                    content = filepath.read_text(encoding="utf-8")
                    data[key] = self._load_tsv(content)
                else:
                    # Try JSON fixture format
                    json_path = path / f"{key}.json"
                    if json_path.exists():
                        data[key] = json.loads(json_path.read_text(encoding="utf-8"))
                    else:
                        data[key] = []
        else:
            raise ValueError(f"snapshot_path must be a .zip file or directory: {path}")

        self._data_cache = data
        return data

    def _build_occupation_text(
        self,
        occ: Dict[str, str],
        skills: List[Dict[str, str]],
        tasks: List[Dict[str, str]],
    ) -> str:
        """Build rich text description for an occupation."""
        parts = []

        # Title and description
        title = occ.get("Title", occ.get("title", "Unknown"))
        desc = occ.get("Description", occ.get("description", ""))
        parts.append(f"{title}: {desc}")

        # Skills (filtered by this occupation)
        onet_code = occ.get("O*NET-SOC Code", occ.get("onet_soc_code", ""))
        if self.include_skills and skills:
            occ_skills = [
                s for s in skills
                if s.get("O*NET-SOC Code", s.get("onet_soc_code", "")) == onet_code
            ]
            # Sort by importance/level and take top skills
            skill_names = []
            for s in occ_skills[:10]:
                name = s.get("Element Name", s.get("element_name", ""))
                if name:
                    skill_names.append(name)
            if skill_names:
                parts.append(f"Key skills: {', '.join(skill_names)}.")

        # Tasks
        if self.include_tasks and tasks:
            occ_tasks = [
                t for t in tasks
                if t.get("O*NET-SOC Code", t.get("onet_soc_code", "")) == onet_code
            ]
            task_texts = []
            for t in occ_tasks[:5]:
                task = t.get("Task", t.get("task", ""))
                if task:
                    task_texts.append(task)
            if task_texts:
                parts.append("Typical tasks: " + " ".join(task_texts))

        return normalize_text(" ".join(parts))

    def fetch(self) -> Iterator[Artifact]:
        """Fetch occupation artifacts from O*NET data."""
        data = self._load_from_snapshot()
        occupations = data.get("occupations", [])
        skills = data.get("skills", [])
        tasks = data.get("tasks", [])

        source_url = f"{ONET_BASE_URL}/db_{ONET_VERSION}.zip"
        if self.snapshot_path:
            source_url = f"file://{self.snapshot_path}"

        count = 0
        for occ in occupations:
            if self.max_occupations and count >= self.max_occupations:
                break

            onet_code = occ.get("O*NET-SOC Code", occ.get("onet_soc_code", ""))
            title = occ.get("Title", occ.get("title", "Unknown"))

            if not onet_code:
                continue

            text = self._build_occupation_text(occ, skills, tasks)
            prov = create_provenance(
                source_url=source_url,
                raw_text=text,
                snapshot_path=self.snapshot_path,
                onet_version=ONET_VERSION,
                onet_soc_code=onet_code,
            )

            yield Artifact(
                artifact_id=f"onet:{onet_code}",
                type=ArtifactType.JOB_ROLE,
                source_url=prov.source_url,
                source_hash=prov.raw_text_hash,
                timestamp=prov.fetched_at,
                text=text,
                metadata={
                    "onet_soc_code": onet_code,
                    "title": title,
                    "provenance": prov.to_dict(),
                },
            )
            count += 1


def create_onet_fixture(output_dir: Path, num_occupations: int = 20) -> None:
    """Create a minimal O*NET fixture for testing.

    This creates sample data that mimics O*NET structure for tests.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample occupations (based on real O*NET structure)
    occupations = [
        {
            "onet_soc_code": "15-2051.00",
            "title": "Data Scientists",
            "description": "Develop and implement methods to collect, process, and analyze large amounts of data. Create predictive models and machine-learning algorithms.",
        },
        {
            "onet_soc_code": "15-1252.00",
            "title": "Software Developers",
            "description": "Research, design, and develop computer and network software or specialized utility programs. Analyze user needs.",
        },
        {
            "onet_soc_code": "15-2031.00",
            "title": "Operations Research Analysts",
            "description": "Formulate and apply mathematical modeling methods to develop and interpret information for management decisions.",
        },
        {
            "onet_soc_code": "17-2199.09",
            "title": "Nanosystems Engineers",
            "description": "Design and develop nanoscale systems and devices with applications in electronics, photonics, biotechnology, or materials science.",
        },
        {
            "onet_soc_code": "19-3011.00",
            "title": "Economists",
            "description": "Conduct research, prepare reports, and formulate plans to solve economic problems. May analyze economic forecasts.",
        },
        {
            "onet_soc_code": "15-1211.00",
            "title": "Computer Systems Analysts",
            "description": "Analyze science, engineering, business, and other data processing problems to develop solutions and automate tasks.",
        },
        {
            "onet_soc_code": "15-1299.08",
            "title": "Computer Systems Engineers/Architects",
            "description": "Design and develop solutions to computer hardware and software problems. Analyze science, engineering, business.",
        },
        {
            "onet_soc_code": "15-2041.00",
            "title": "Statisticians",
            "description": "Develop or apply mathematical or statistical theory and methods to collect, organize, interpret, and summarize data.",
        },
        {
            "onet_soc_code": "11-3021.00",
            "title": "Computer and Information Systems Managers",
            "description": "Plan, direct, or coordinate activities in electronic data processing, information systems, or computer programming.",
        },
        {
            "onet_soc_code": "15-1212.00",
            "title": "Information Security Analysts",
            "description": "Plan, implement, upgrade, or monitor security measures for the protection of computer networks and information.",
        },
    ]

    # Sample skills
    skills = [
        {"onet_soc_code": "15-2051.00", "element_name": "Programming"},
        {"onet_soc_code": "15-2051.00", "element_name": "Mathematics"},
        {"onet_soc_code": "15-2051.00", "element_name": "Critical Thinking"},
        {"onet_soc_code": "15-2051.00", "element_name": "Complex Problem Solving"},
        {"onet_soc_code": "15-1252.00", "element_name": "Programming"},
        {"onet_soc_code": "15-1252.00", "element_name": "Systems Analysis"},
        {"onet_soc_code": "15-1252.00", "element_name": "Quality Control Analysis"},
        {"onet_soc_code": "15-2031.00", "element_name": "Mathematics"},
        {"onet_soc_code": "15-2031.00", "element_name": "Operations Analysis"},
        {"onet_soc_code": "15-1211.00", "element_name": "Systems Evaluation"},
    ]

    # Sample tasks
    tasks = [
        {"onet_soc_code": "15-2051.00", "task": "Apply data mining, machine learning, and statistical techniques to solve business problems."},
        {"onet_soc_code": "15-2051.00", "task": "Develop custom data models and algorithms."},
        {"onet_soc_code": "15-1252.00", "task": "Modify existing software to correct errors or improve performance."},
        {"onet_soc_code": "15-1252.00", "task": "Design and implement software applications."},
        {"onet_soc_code": "15-2031.00", "task": "Formulate mathematical or simulation models of problems."},
    ]

    # Write fixture files
    (output_dir / "occupations.json").write_text(
        json.dumps(occupations, indent=2), encoding="utf-8"
    )
    (output_dir / "skills.json").write_text(
        json.dumps(skills, indent=2), encoding="utf-8"
    )
    (output_dir / "tasks.json").write_text(
        json.dumps(tasks, indent=2), encoding="utf-8"
    )

    # Write provenance metadata
    meta = {
        "source": "O*NET OnLine - synthetic fixture for testing",
        "onet_version": ONET_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "license": "Public Domain (O*NET data is in the public domain)",
        "note": "This is a minimal fixture for unit tests, not full O*NET data.",
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
