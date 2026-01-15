"""Diagonal lens - element-wise scaling transformation.

The diagonal lens scales each embedding dimension independently:
L(v) = diag(w) * v = [w_1*v_1, w_2*v_2, ..., w_d*v_d]

This provides:
1. Interpretability: w_i represents dimension i's importance
2. Efficiency: O(d) computation vs O(d²) for full matrix
3. Regularization: fewer parameters than full transformation

Weight interpretation:
- w_i > 1: Amplify dimension i (more important for this stakeholder)
- w_i = 1: Keep dimension i as-is (neutral)
- w_i < 1: Dampen dimension i (less important)
- w_i = 0: Completely ignore dimension i
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from ads_core.lenses.base import Lens


class DiagonalLens(Lens):
    """Diagonal lens with element-wise weight scaling.

    L(v) = diag(w) * v

    Each weight w_i controls the importance of embedding dimension i
    for this stakeholder's perspective.

    Example:
        >>> weights = np.array([2.0, 1.0, 0.5])  # Amplify dim 0, dampen dim 2
        >>> lens = DiagonalLens("diag:market", weights)
        >>> v = np.array([1.0, 1.0, 1.0])
        >>> lens.transform(v)
        array([2. , 1. , 0.5])
    """

    def __init__(
        self,
        lens_id: str,
        weights: np.ndarray,
        normalize: bool = False,
    ):
        """Initialize diagonal lens.

        Args:
            lens_id: Unique identifier (e.g., "diag:market")
            weights: Weight vector of shape [d]
            normalize: If True, normalize weights to mean=1
        """
        self.lens_id = lens_id
        w = weights.astype(np.float32)

        if normalize and w.mean() > 1e-8:
            w = w / w.mean()

        self.w = w
        self._dim = len(w)

    @property
    def weights(self) -> np.ndarray:
        """Get weight vector."""
        return self.w

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    def transform(self, v: np.ndarray) -> np.ndarray:
        """Apply diagonal scaling.

        Args:
            v: Input vector of shape [d] or [n, d]

        Returns:
            Scaled vector(s) of same shape
        """
        return v * self.w

    def get_top_dimensions(self, k: int = 10) -> Dict[str, Any]:
        """Get the k most important dimensions.

        Args:
            k: Number of dimensions to return

        Returns:
            Dict with indices and weights of top dimensions
        """
        idx = np.argsort(self.w)[-k:][::-1]
        return {
            "indices": idx.tolist(),
            "weights": self.w[idx].tolist(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize lens metadata."""
        return {
            "lens_id": self.lens_id,
            "type": "DiagonalLens",
            "mode": "diagonal",
            "dim": self._dim,
            "weight_mean": float(self.w.mean()),
            "weight_std": float(self.w.std()),
            "weight_min": float(self.w.min()),
            "weight_max": float(self.w.max()),
        }


def create_uniform_lens(lens_id: str, dim: int) -> DiagonalLens:
    """Create a diagonal lens with uniform weights (equivalent to identity).

    Args:
        lens_id: Lens identifier
        dim: Embedding dimension

    Returns:
        DiagonalLens with all weights = 1.0
    """
    return DiagonalLens(lens_id, np.ones(dim, dtype=np.float32))


def create_random_lens(
    lens_id: str,
    dim: int,
    seed: int = 42,
    scale: float = 1.0,
) -> DiagonalLens:
    """Create a diagonal lens with random weights (for ablation).

    Args:
        lens_id: Lens identifier
        dim: Embedding dimension
        seed: Random seed for reproducibility
        scale: Scale factor for weights

    Returns:
        DiagonalLens with random positive weights
    """
    rng = np.random.RandomState(seed)
    # Use exponential to get positive weights with some variance
    weights = rng.exponential(scale, size=dim).astype(np.float32)
    return DiagonalLens(lens_id, weights, normalize=True)
