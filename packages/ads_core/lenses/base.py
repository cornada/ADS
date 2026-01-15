"""Base lens class for stakeholder-specific transformations.

Lenses are the core ADS novelty: L_a(v) = W_a * v

Each stakeholder (market, mission, university, learner) can have a different lens
that emphasizes different dimensions of the embedding space. This allows the same
course embedding to be evaluated differently from each stakeholder's perspective.

Lens Types:
- IdentityLens: L(v) = v (baseline, no transformation)
- DiagonalLens: L(v) = diag(w) * v (element-wise scaling)
- LearnedDiagonalLens: Weights learned via one-vs-rest classification

The diagonal form is interpretable: w_i represents the importance of dimension i
for this stakeholder. Higher weights amplify that dimension's contribution to
similarity computations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class Lens(ABC):
    """Abstract base class for stakeholder lenses.

    A lens transforms embedding vectors to emphasize dimensions relevant
    to a specific stakeholder's perspective.

    Attributes:
        lens_id: Unique identifier for this lens (e.g., "identity:market")
    """

    lens_id: str

    @abstractmethod
    def transform(self, v: np.ndarray) -> np.ndarray:
        """Transform an embedding vector through this lens.

        Args:
            v: Input embedding vector of shape [d] or [n, d]

        Returns:
            Transformed vector(s) of same shape
        """
        raise NotImplementedError

    def explain(self, v: np.ndarray, top_k: int = 10) -> Dict[str, Any]:
        """Explain which dimensions are most activated after transformation.

        Args:
            v: Input embedding vector
            top_k: Number of top dimensions to return

        Returns:
            Dict with top_dims (indices) and values (magnitudes)
        """
        vv = self.transform(v)
        idx = np.argsort(np.abs(vv))[-top_k:][::-1]
        return {
            "lens_id": self.lens_id,
            "top_dims": idx.tolist(),
            "values": vv[idx].tolist(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize lens metadata."""
        return {
            "lens_id": self.lens_id,
            "type": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lens_id={self.lens_id!r})"
