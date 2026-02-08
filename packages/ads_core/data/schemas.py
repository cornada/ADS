from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ArtifactType(str, Enum):
    MISSION = "MISSION"
    COURSE = "COURSE"
    SYLLABUS = "SYLLABUS"
    LAB = "LAB"
    PAPER_ABSTRACT = "PAPER_ABSTRACT"
    JOB_ROLE = "JOB_ROLE"
    SKILL = "SKILL"
    LEARNER_PROFILE = "LEARNER_PROFILE"
    OUTCOME_MAJOR = "OUTCOME_MAJOR"  # Career outcomes by major (e.g., FDS data)
    OUTCOME_SUMMARY = "OUTCOME_SUMMARY"  # Aggregated outcomes summary (e.g., ASU reports)
    SURVEY_SCHEMA = "SURVEY_SCHEMA"  # Survey instrument schema (e.g., FDS questions)


class Artifact(BaseModel):
    artifact_id: str
    type: ArtifactType
    source_url: Optional[str] = None
    source_hash: Optional[str] = None
    timestamp: Optional[datetime] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingRecord(BaseModel):
    artifact_id: str
    model_id: str
    vector: List[float]
    created_at: datetime


class Stakeholder(BaseModel):
    stakeholder_id: str
    lens_id: str
    target_spec: Dict[str, Any] = Field(default_factory=dict)


class Option(BaseModel):
    option_id: str
    artifact_id: str
    kind: str
    action_spec: Dict[str, Any] = Field(default_factory=dict)


class EvaluationVector(BaseModel):
    option_id: str
    objectives: Dict[str, float]
    feasible: bool = True
    violations: Dict[str, Any] = Field(default_factory=dict)


class Pathway(BaseModel):
    pathway_id: str
    option_ids: List[str]
    objectives: Dict[str, float]
    feasible: bool = True
    violations: Dict[str, Any] = Field(default_factory=dict)
