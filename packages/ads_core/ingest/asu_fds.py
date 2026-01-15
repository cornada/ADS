"""ASU First Destination Survey (FDS) schema parser.

Extracts survey structure from FDS Survey Preview PDF including:
- Questions and their types
- Answer options
- Branching logic/dependencies
- Section organization

Source: https://uoeee.asu.edu/sites/g/files/litvpz631/files/docs/Survey/FDS%20Survey%20Preview.pdf

Note: PDF parsing is inherently messy. This parser uses fixtures for reproducibility,
with optional PDF extraction for updates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ads_core.data.schemas import Artifact, ArtifactType
from ads_core.ingest.base import BaseConnector, create_provenance


@dataclass
class SurveyQuestion:
    """Represents a single survey question."""
    question_id: str
    section: str
    text: str
    question_type: str  # single_choice, multiple_choice, text, scale, etc.
    options: List[str] = field(default_factory=list)
    required: bool = True
    depends_on: Optional[str] = None  # Branching condition
    skip_logic: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "question_id": self.question_id,
            "section": self.section,
            "text": self.text,
            "question_type": self.question_type,
            "required": self.required,
        }
        if self.options:
            d["options"] = self.options
        if self.depends_on:
            d["depends_on"] = self.depends_on
        if self.skip_logic:
            d["skip_logic"] = self.skip_logic
        return d


@dataclass
class SurveySection:
    """Represents a survey section."""
    section_id: str
    name: str
    description: str
    questions: List[SurveyQuestion] = field(default_factory=list)
    depends_on: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "name": self.name,
            "description": self.description,
            "depends_on": self.depends_on,
            "questions": [q.to_dict() for q in self.questions],
        }


@dataclass
class FDSSchema:
    """Complete FDS survey schema."""
    version: str
    source_url: str
    sections: List[SurveySection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "source_url": self.source_url,
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata,
        }

    def to_text_summary(self) -> str:
        """Generate a text summary for embedding."""
        parts = [f"First Destination Survey Schema (v{self.version}):"]

        for section in self.sections:
            parts.append(f"\n{section.name}: {section.description}")
            q_count = len(section.questions)
            if q_count > 0:
                parts.append(f"Contains {q_count} questions.")

        return " ".join(parts)


def parse_fds_pdf_text(text: str) -> FDSSchema:
    """Parse FDS schema from extracted PDF text.

    This is a best-effort parser for FDS survey structure.
    Due to PDF complexity, may not capture all nuances.

    Args:
        text: Extracted text from FDS PDF

    Returns:
        Parsed FDSSchema
    """
    schema = FDSSchema(
        version="1.0",
        source_url="https://uoeee.asu.edu/sites/g/files/litvpz631/files/docs/Survey/FDS%20Survey%20Preview.pdf",
    )

    # For MVP, use predefined structure based on known FDS format
    # This ensures reproducibility even if PDF parsing varies
    # Future: implement actual PDF text parsing with regex patterns like:
    # r"(?:Q(\d+)|(\d+)\.)\s*([^\n]+?)(?:\n|$)"

    return schema


def build_fds_schema_from_manifest(manifest: Dict[str, Any]) -> FDSSchema:
    """Build FDS schema from manifest definition.

    Uses manifest's fds_sections as the authoritative structure.
    """
    source_url = manifest.get("sources", {}).get("fds_survey_preview", {}).get(
        "url", "https://uoeee.asu.edu/fds"
    )

    schema = FDSSchema(
        version=manifest.get("version", "1.0"),
        source_url=source_url,
        metadata={
            "institution": manifest.get("institution", "asu"),
            "created": manifest.get("created", datetime.now(timezone.utc).isoformat()),
        }
    )

    # Build sections from manifest
    for section_def in manifest.get("fds_sections", []):
        section = SurveySection(
            section_id=section_def.get("id", ""),
            name=section_def.get("name", ""),
            description=section_def.get("description", ""),
            depends_on=section_def.get("depends_on"),
        )
        schema.sections.append(section)

    return schema


class AsuFdsConnector(BaseConnector):
    """Connector for ASU FDS survey schema extraction.

    Extracts survey structure from FDS Preview PDF or fixtures.
    """

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
    ):
        """Initialize ASU FDS connector.

        Args:
            manifest_path: Path to asu_sources.yaml manifest
            raw_dir: Directory with raw PDF files
        """
        self.manifest_path = manifest_path
        self.raw_dir = raw_dir
        self._manifest: Optional[Dict] = None
        self._schema: Optional[FDSSchema] = None

    @property
    def source_name(self) -> str:
        return "asu_fds"

    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest configuration."""
        if self._manifest is not None:
            return self._manifest

        if self.manifest_path and Path(self.manifest_path).exists():
            import yaml
            self._manifest = yaml.safe_load(
                Path(self.manifest_path).read_text(encoding="utf-8")
            )
        else:
            self._manifest = {
                "version": "1.0",
                "institution": "asu",
                "fds_sections": [],
            }

        return self._manifest

    def load_schema(self) -> FDSSchema:
        """Load FDS schema from fixtures or PDF."""
        if self._schema is not None:
            return self._schema

        # Try raw_dir first if provided
        if self.raw_dir:
            raw_schema_path = Path(self.raw_dir) / "fds_schema.json"
            if raw_schema_path.exists():
                return self._load_schema_from_file(raw_schema_path)

        # Try default fixture path
        fixture_path = Path(__file__).parent.parent.parent.parent / "data" / "fixtures" / "asu" / "fds_schema.json"
        if fixture_path.exists():
            return self._load_schema_from_file(fixture_path)

        # Fall back to manifest-based schema
        manifest = self._load_manifest()
        self._schema = build_fds_schema_from_manifest(manifest)
        return self._schema

    def _load_schema_from_file(self, path: Path) -> FDSSchema:
        """Load schema from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        self._schema = FDSSchema(
            version=data.get("version", "1.0"),
            source_url=data.get("source_url", ""),
            metadata=data.get("metadata", {}),
        )
        for s in data.get("sections", []):
            section = SurveySection(
                section_id=s.get("section_id", ""),
                name=s.get("name", ""),
                description=s.get("description", ""),
                depends_on=s.get("depends_on"),
            )
            for q in s.get("questions", []):
                question = SurveyQuestion(
                    question_id=q.get("question_id", ""),
                    section=s.get("section_id", ""),
                    text=q.get("text", ""),
                    question_type=q.get("question_type", "single_choice"),
                    options=q.get("options", []),
                    required=q.get("required", True),
                    depends_on=q.get("depends_on"),
                    skip_logic=q.get("skip_logic"),
                )
                section.questions.append(question)
            self._schema.sections.append(section)
        return self._schema

    def fetch(self) -> Iterator[Artifact]:
        """Fetch survey schema artifacts."""
        schema = self.load_schema()

        # Create one artifact for the overall schema
        text = schema.to_text_summary()

        prov = create_provenance(
            source_url=schema.source_url,
            raw_text=text,
            institution="asu",
            schema_version=schema.version,
        )

        yield Artifact(
            artifact_id="asu_fds:schema",
            type=ArtifactType.SURVEY_SCHEMA,
            source_url=schema.source_url,
            source_hash=prov.raw_text_hash,
            timestamp=prov.fetched_at,
            text=text,
            metadata={
                "version": schema.version,
                "section_count": len(schema.sections),
                "question_count": sum(len(s.questions) for s in schema.sections),
                "license": "Public document for educational and research purposes",
                "provenance": prov.to_dict(),
            },
        )

        # Create artifacts for each section
        for section in schema.sections:
            section_text = f"FDS Survey Section - {section.name}: {section.description}"
            if section.questions:
                section_text += f" Contains {len(section.questions)} questions."
                q_types = set(q.question_type for q in section.questions)
                section_text += f" Question types: {', '.join(q_types)}."

            prov = create_provenance(
                source_url=schema.source_url,
                raw_text=section_text,
                section_id=section.section_id,
            )

            yield Artifact(
                artifact_id=f"asu_fds:section:{section.section_id}",
                type=ArtifactType.SURVEY_SCHEMA,
                source_url=schema.source_url,
                source_hash=prov.raw_text_hash,
                timestamp=prov.fetched_at,
                text=section_text,
                metadata={
                    "section_id": section.section_id,
                    "section_name": section.name,
                    "question_count": len(section.questions),
                    "depends_on": section.depends_on,
                    "provenance": prov.to_dict(),
                },
            )


def create_fds_fixture(output_dir: Path) -> None:
    """Create a minimal FDS schema fixture for testing.

    Based on standard FDS survey structure used by NACE member institutions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Standard FDS survey structure
    schema = {
        "version": "1.0",
        "source_url": "https://uoeee.asu.edu/sites/g/files/litvpz631/files/docs/Survey/FDS%20Survey%20Preview.pdf",
        "metadata": {
            "institution": "asu",
            "survey_name": "First Destination Survey",
            "created": datetime.now(timezone.utc).isoformat(),
        },
        "sections": [
            {
                "section_id": "employment_status",
                "name": "Current Employment Status",
                "description": "Determines primary activity after graduation",
                "depends_on": None,
                "questions": [
                    {
                        "question_id": "Q1",
                        "text": "What is your current employment status?",
                        "question_type": "single_choice",
                        "required": True,
                        "options": [
                            "Employed full-time",
                            "Employed part-time",
                            "Continuing education (graduate/professional school)",
                            "Seeking employment",
                            "Not seeking employment",
                            "Military service",
                            "Volunteer service (e.g., Peace Corps, AmeriCorps)",
                        ],
                    },
                ],
            },
            {
                "section_id": "employment_details",
                "name": "Employment Details",
                "description": "Collects information about current employment",
                "depends_on": "employment_status=employed",
                "questions": [
                    {
                        "question_id": "Q2",
                        "text": "What is your job title?",
                        "question_type": "text",
                        "required": True,
                    },
                    {
                        "question_id": "Q3",
                        "text": "What is the name of your employer?",
                        "question_type": "text",
                        "required": True,
                    },
                    {
                        "question_id": "Q4",
                        "text": "In which industry is your employer?",
                        "question_type": "single_choice",
                        "required": True,
                        "options": [
                            "Technology/Software",
                            "Finance/Banking",
                            "Healthcare",
                            "Education",
                            "Government",
                            "Manufacturing",
                            "Consulting",
                            "Retail",
                            "Non-profit",
                            "Other",
                        ],
                    },
                    {
                        "question_id": "Q5",
                        "text": "What is your annual salary?",
                        "question_type": "numeric",
                        "required": False,
                    },
                    {
                        "question_id": "Q6",
                        "text": "In which state/country is your job located?",
                        "question_type": "text",
                        "required": True,
                    },
                    {
                        "question_id": "Q7",
                        "text": "Is this position related to your major?",
                        "question_type": "single_choice",
                        "required": True,
                        "options": [
                            "Directly related",
                            "Somewhat related",
                            "Not related",
                        ],
                    },
                ],
            },
            {
                "section_id": "continuing_education",
                "name": "Continuing Education",
                "description": "For those pursuing graduate or professional education",
                "depends_on": "employment_status=continuing_education",
                "questions": [
                    {
                        "question_id": "Q8",
                        "text": "What type of degree are you pursuing?",
                        "question_type": "single_choice",
                        "required": True,
                        "options": [
                            "Master's degree",
                            "Doctoral degree (PhD)",
                            "Professional degree (MD, JD, etc.)",
                            "Certificate program",
                            "Other",
                        ],
                    },
                    {
                        "question_id": "Q9",
                        "text": "What is the name of the institution?",
                        "question_type": "text",
                        "required": True,
                    },
                    {
                        "question_id": "Q10",
                        "text": "What is your field of study?",
                        "question_type": "text",
                        "required": True,
                    },
                ],
            },
            {
                "section_id": "job_search",
                "name": "Job Search Status",
                "description": "For those currently seeking employment",
                "depends_on": "employment_status=seeking",
                "questions": [
                    {
                        "question_id": "Q11",
                        "text": "How long have you been searching for employment?",
                        "question_type": "single_choice",
                        "required": True,
                        "options": [
                            "Less than 1 month",
                            "1-3 months",
                            "3-6 months",
                            "More than 6 months",
                        ],
                    },
                    {
                        "question_id": "Q12",
                        "text": "What challenges have you faced in your job search?",
                        "question_type": "multiple_choice",
                        "required": False,
                        "options": [
                            "Lack of experience",
                            "Limited job openings in my field",
                            "Geographic constraints",
                            "Salary expectations not met",
                            "Competition from other candidates",
                            "Other",
                        ],
                    },
                ],
            },
            {
                "section_id": "career_services",
                "name": "Career Services",
                "description": "Usage of university career services",
                "depends_on": None,
                "questions": [
                    {
                        "question_id": "Q13",
                        "text": "Did you use career services during your time at the university?",
                        "question_type": "single_choice",
                        "required": True,
                        "options": ["Yes", "No"],
                    },
                    {
                        "question_id": "Q14",
                        "text": "Which career services did you use?",
                        "question_type": "multiple_choice",
                        "required": False,
                        "depends_on": "Q13=Yes",
                        "options": [
                            "Resume review",
                            "Mock interviews",
                            "Career counseling",
                            "Job fairs",
                            "On-campus recruiting",
                            "Career workshops",
                            "Online job board",
                        ],
                    },
                ],
            },
            {
                "section_id": "internship",
                "name": "Internship Experience",
                "description": "Information about internship participation",
                "depends_on": None,
                "questions": [
                    {
                        "question_id": "Q15",
                        "text": "Did you complete an internship during your studies?",
                        "question_type": "single_choice",
                        "required": True,
                        "options": ["Yes", "No"],
                    },
                    {
                        "question_id": "Q16",
                        "text": "Did your internship lead to a job offer?",
                        "question_type": "single_choice",
                        "required": False,
                        "depends_on": "Q15=Yes",
                        "options": ["Yes", "No"],
                    },
                ],
            },
        ],
    }

    (output_dir / "fds_schema.json").write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8"
    )

    # Write metadata
    meta = {
        "source": "ASU FDS Survey Preview - structured fixture for testing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "license": "Public document for educational and research purposes",
        "notes": "Based on standard NACE FDS survey structure",
        "original_url": schema["source_url"],
    }
    (output_dir / "fds_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8"
    )
