from __future__ import annotations

from typing import List
from datetime import datetime, timezone

from ads_core.data.schemas import Artifact, ArtifactType


def build_toy_artifacts() -> List[Artifact]:
    now = datetime.now(timezone.utc)
    artifacts: List[Artifact] = []

    # Missions
    artifacts.append(Artifact(
        artifact_id="mission:dept_ai",
        type=ArtifactType.MISSION,
        source_url="toy://mission/dept_ai",
        source_hash="toy",
        timestamp=now,
        text="Our mission is to advance responsible and human-centered AI, combining theory and practice."
    ))

    # Market job roles
    artifacts.append(Artifact(
        artifact_id="job:ml_engineer",
        type=ArtifactType.JOB_ROLE,
        source_url="toy://job/ml_engineer",
        source_hash="toy",
        timestamp=now,
        text="Machine learning engineer: deploys models, builds data pipelines, optimizes performance, works with distributed systems."
    ))
    artifacts.append(Artifact(
        artifact_id="job:ai_policy",
        type=ArtifactType.JOB_ROLE,
        source_url="toy://job/ai_policy",
        source_hash="toy",
        timestamp=now,
        text="AI policy analyst: evaluates societal impacts, governance, fairness, regulation, and ethical risks."
    ))

    # Courses (options)
    courses = [
        ("course:dl", "Deep Learning Fundamentals: optimization, neural networks, training stability, generalization."),
        ("course:ethics", "AI Ethics and Policy: fairness, accountability, transparency, governance, risk assessment."),
        ("course:systems", "Scalable Systems for ML: distributed data processing, deployment, monitoring, reliability."),
        ("course:stats", "Statistical Learning Theory: bias-variance, regularization, uncertainty, inference."),
        ("course:hci", "Human-AI Interaction: user studies, interpretability, contestability, interaction design."),
    ]
    for cid, text in courses:
        artifacts.append(Artifact(
            artifact_id=cid,
            type=ArtifactType.COURSE,
            source_url=f"toy://course/{cid}",
            source_hash="toy",
            timestamp=now,
            text=text,
        ))

    # Learner profile
    artifacts.append(Artifact(
        artifact_id="learner:demo",
        type=ArtifactType.LEARNER_PROFILE,
        source_url="toy://learner/demo",
        source_hash="toy",
        timestamp=now,
        text="I want to build responsible AI systems and work on ML engineering in real products."
    ))

    return artifacts
