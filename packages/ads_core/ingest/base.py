"""Base classes for data ingestion connectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ads_core.data.schemas import Artifact, ArtifactType
from ads_core.utils.hashing import sha256_text


@dataclass
class ProvenanceInfo:
    """Tracks provenance for reproducibility."""

    source_url: str
    fetched_at: datetime
    raw_text_hash: str
    snapshot_path: Optional[Path] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_url": self.source_url,
            "fetched_at": self.fetched_at.isoformat(),
            "raw_text_hash": self.raw_text_hash,
            "snapshot_path": str(self.snapshot_path) if self.snapshot_path else None,
            **self.extra,
        }


def create_provenance(
    source_url: str,
    raw_text: str,
    snapshot_path: Optional[Path] = None,
    **extra: Any,
) -> ProvenanceInfo:
    """Create a provenance record with current timestamp and hash."""
    return ProvenanceInfo(
        source_url=source_url,
        fetched_at=datetime.now(timezone.utc),
        raw_text_hash=sha256_text(raw_text),
        snapshot_path=snapshot_path,
        extra=extra,
    )


class BaseConnector(ABC):
    """Abstract base class for data source connectors."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique name identifying this data source."""
        ...

    @abstractmethod
    def fetch(self) -> Iterator[Artifact]:
        """Fetch and yield artifacts from the data source."""
        ...

    def fetch_all(self) -> List[Artifact]:
        """Fetch all artifacts as a list."""
        return list(self.fetch())


def normalize_text(text: str) -> str:
    """Normalize whitespace in text for consistent hashing."""
    return " ".join(text.split())
