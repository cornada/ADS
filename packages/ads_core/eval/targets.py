"""Target set builders for objective evaluation.

Target sets represent the "ideal" directions in embedding space for each stakeholder:
- market: centroid of job role embeddings (what employers want)
- mission: centroid of university mission statements (institutional values)
- university: centroid of course embeddings (what the curriculum offers)
- learner: the learner's profile embedding (personal goals)

Two main representations:
1. Centroid: single normalized vector (mean of set)
2. Cloud: set of points with distance aggregation (min, mean, max)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from ads_core.eval.distances import cosine_distance, cosine_similarity


class AggregationMethod(str, Enum):
    """How to aggregate distances to a cloud of points."""
    MIN = "min"      # Closest point (optimistic)
    MEAN = "mean"    # Average distance
    MAX = "max"      # Farthest point (pessimistic)
    MEDIAN = "median"


class TargetSet(ABC):
    """Abstract base class for target sets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this target set (e.g., 'market', 'mission')."""
        ...

    @abstractmethod
    def distance(self, v: np.ndarray) -> float:
        """Compute distance from vector v to this target set.

        Convention: lower is better (closer to target).
        Returns value in [0, 2] for cosine distance.
        """
        ...

    @abstractmethod
    def similarity(self, v: np.ndarray) -> float:
        """Compute similarity from vector v to this target set.

        Convention: higher is better (more aligned).
        Returns value in [-1, 1] for cosine similarity.
        """
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize target set metadata."""
        ...


@dataclass
class CentroidTarget(TargetSet):
    """Target set represented as a single centroid vector.

    The centroid is the normalized mean of the source vectors.
    This is efficient and works well when the target set is unimodal.
    """

    _name: str
    centroid: np.ndarray
    source_count: int = 0
    _metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self._name

    def distance(self, v: np.ndarray) -> float:
        """Cosine distance to centroid."""
        return cosine_distance(v, self.centroid)

    def similarity(self, v: np.ndarray) -> float:
        """Cosine similarity to centroid."""
        return cosine_similarity(v, self.centroid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "type": "centroid",
            "source_count": self.source_count,
            "centroid_norm": float(np.linalg.norm(self.centroid)),
            **self._metadata,
        }


@dataclass
class CloudTarget(TargetSet):
    """Target set represented as a cloud of points.

    Useful when the target is multimodal (e.g., diverse job roles).
    Distance can be aggregated as min, mean, max, or median.
    """

    _name: str
    points: np.ndarray  # [n, d]
    aggregation: AggregationMethod = AggregationMethod.MIN
    _metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return self.points.shape[0]

    def _compute_distances(self, v: np.ndarray) -> np.ndarray:
        """Compute distances to all points."""
        distances = np.array([cosine_distance(v, p) for p in self.points])
        return distances

    def distance(self, v: np.ndarray) -> float:
        """Aggregated distance to cloud."""
        if self.size == 0:
            return 2.0  # Max cosine distance

        distances = self._compute_distances(v)

        if self.aggregation == AggregationMethod.MIN:
            return float(np.min(distances))
        elif self.aggregation == AggregationMethod.MEAN:
            return float(np.mean(distances))
        elif self.aggregation == AggregationMethod.MAX:
            return float(np.max(distances))
        elif self.aggregation == AggregationMethod.MEDIAN:
            return float(np.median(distances))
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def similarity(self, v: np.ndarray) -> float:
        """Convert distance to similarity."""
        return 1.0 - self.distance(v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "type": "cloud",
            "size": self.size,
            "aggregation": self.aggregation.value,
            **self._metadata,
        }


def build_centroid_target(
    name: str,
    vectors: List[np.ndarray],
    metadata: Optional[Dict[str, Any]] = None,
) -> CentroidTarget:
    """Build a centroid target from a list of vectors.

    Args:
        name: Target set name (e.g., "market")
        vectors: List of embedding vectors
        metadata: Optional metadata dict

    Returns:
        CentroidTarget with normalized centroid
    """
    if len(vectors) == 0:
        raise ValueError(f"Cannot build centroid target '{name}' from empty vectors")

    # Stack and compute mean
    stacked = np.stack(vectors, axis=0)
    mean = stacked.mean(axis=0)

    # Normalize
    norm = np.linalg.norm(mean)
    if norm < 1e-8:
        # Edge case: vectors cancel out
        centroid = mean
    else:
        centroid = mean / norm

    return CentroidTarget(
        _name=name,
        centroid=centroid.astype(np.float32),
        source_count=len(vectors),
        _metadata=metadata or {},
    )


def build_cloud_target(
    name: str,
    vectors: List[np.ndarray],
    aggregation: AggregationMethod = AggregationMethod.MIN,
    metadata: Optional[Dict[str, Any]] = None,
) -> CloudTarget:
    """Build a cloud target from a list of vectors.

    Args:
        name: Target set name
        vectors: List of embedding vectors
        aggregation: How to aggregate distances
        metadata: Optional metadata dict

    Returns:
        CloudTarget with all points
    """
    if len(vectors) == 0:
        points = np.zeros((0, 1), dtype=np.float32)
    else:
        points = np.stack(vectors, axis=0).astype(np.float32)

    return CloudTarget(
        _name=name,
        points=points,
        aggregation=aggregation,
        _metadata=metadata or {},
    )


def build_single_point_target(
    name: str,
    vector: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> CentroidTarget:
    """Build a target from a single point (e.g., learner profile).

    Args:
        name: Target set name
        vector: Single embedding vector
        metadata: Optional metadata dict

    Returns:
        CentroidTarget with the normalized point
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        normalized = vector
    else:
        normalized = vector / norm

    return CentroidTarget(
        _name=name,
        centroid=normalized.astype(np.float32),
        source_count=1,
        _metadata=metadata or {"type": "single_point"},
    )


@dataclass
class TargetSetCollection:
    """Collection of target sets for multi-objective evaluation.

    This class manages the four standard ADS targets:
    - market: alignment with job market demands
    - mission: alignment with institutional mission
    - university: alignment with curriculum
    - learner: alignment with personal goals
    """

    targets: Dict[str, TargetSet] = field(default_factory=dict)

    def add(self, target: TargetSet) -> None:
        """Add a target set."""
        self.targets[target.name] = target

    def get(self, name: str) -> Optional[TargetSet]:
        """Get a target set by name."""
        return self.targets.get(name)

    def names(self) -> List[str]:
        """Get all target names."""
        return list(self.targets.keys())

    def evaluate_distances(self, v: np.ndarray) -> Dict[str, float]:
        """Compute distances to all targets.

        Args:
            v: Option embedding vector

        Returns:
            Dict mapping target name to distance (lower is better)
        """
        return {name: target.distance(v) for name, target in self.targets.items()}

    def evaluate_similarities(self, v: np.ndarray) -> Dict[str, float]:
        """Compute similarities to all targets.

        Args:
            v: Option embedding vector

        Returns:
            Dict mapping target name to similarity (higher is better)
        """
        return {name: target.similarity(v) for name, target in self.targets.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize collection metadata."""
        return {
            "targets": {name: t.to_dict() for name, t in self.targets.items()},
            "num_targets": len(self.targets),
        }
