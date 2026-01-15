"""Identity lens - baseline with no transformation.

The identity lens passes vectors through unchanged: L(v) = v

This serves as:
1. A baseline for ablation studies (comparing against diagonal/learned lenses)
2. The default when no stakeholder-specific weighting is needed
3. A sanity check that the system works without lens transformations
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ads_core.lenses.base import Lens


class IdentityLens(Lens):
    """Identity lens that returns vectors unchanged.

    L(v) = v (no transformation)

    This is the simplest lens and serves as a baseline. All embedding
    dimensions contribute equally to similarity computations.

    Example:
        >>> lens = IdentityLens(lens_id="identity:market")
        >>> v = np.array([1.0, 2.0, 3.0])
        >>> lens.transform(v)
        array([1., 2., 3.])
    """

    def __init__(self, lens_id: str = "identity"):
        """Initialize identity lens.

        Args:
            lens_id: Unique identifier (e.g., "identity:market")
        """
        self.lens_id = lens_id

    def transform(self, v: np.ndarray) -> np.ndarray:
        """Return vector unchanged.

        Args:
            v: Input vector of shape [d] or [n, d]

        Returns:
            Same vector (no copy made for efficiency)
        """
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Serialize lens metadata."""
        return {
            "lens_id": self.lens_id,
            "type": "IdentityLens",
            "mode": "identity",
        }
