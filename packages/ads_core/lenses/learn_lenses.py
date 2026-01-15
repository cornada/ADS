"""Learned lens weight estimation via one-vs-rest classification.

The key insight of ADS is that different stakeholders emphasize different
embedding dimensions. This module learns these emphases from data:

Method:
1. Collect embeddings labeled by stakeholder (market, mission, university, learner)
2. For each stakeholder, train a one-vs-rest binary classifier
3. Extract feature importances from the classifier coefficients
4. Use |coef_i| as the lens weight for dimension i

Why Logistic Regression:
- Linear model gives interpretable per-dimension weights
- L2 regularization prevents overfitting on small corpora
- Coefficients directly indicate which dimensions discriminate this stakeholder

Weight Interpretation:
- Higher weight = dimension is more important for this stakeholder
- Weights are normalized to mean 1.0 to preserve embedding scale
- A weight of 2.0 means that dimension is 2x as important as average

Example:
    # Assume we have embeddings and labels
    X = encoder.encode_batch(texts)  # [n, d]
    labels = ["market", "market", "mission", ...]  # stakeholder labels

    learned = learn_one_vs_rest_diagonal_weights(X, labels, ["market", "mission"])
    market_lens = DiagonalLens("learned:market", learned.weights["market"])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from sklearn.linear_model import LogisticRegression

from ads_core.lenses.diagonal import DiagonalLens


@dataclass
class LearnedDiagonalLenses:
    """Container for learned lens weights across stakeholders.

    Attributes:
        weights: Dict mapping stakeholder name to weight vector [d]
        metadata: Training metadata (seed, n_samples, etc.)
    """

    weights: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_lens(self, stakeholder: str, prefix: str = "learned") -> DiagonalLens:
        """Create a DiagonalLens from learned weights.

        Args:
            stakeholder: Stakeholder name (e.g., "market")
            prefix: Lens ID prefix (default "learned")

        Returns:
            DiagonalLens with learned weights
        """
        if stakeholder not in self.weights:
            raise KeyError(f"No learned weights for stakeholder: {stakeholder}")
        return DiagonalLens(
            lens_id=f"{prefix}:{stakeholder}",
            weights=self.weights[stakeholder],
            normalize=False,  # Already normalized during learning
        )

    def get_all_lenses(self, prefix: str = "learned") -> Dict[str, DiagonalLens]:
        """Create DiagonalLens for all stakeholders.

        Args:
            prefix: Lens ID prefix

        Returns:
            Dict mapping stakeholder name to DiagonalLens
        """
        return {s: self.get_lens(s, prefix) for s in self.weights}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "weights": {k: v.tolist() for k, v in self.weights.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedDiagonalLenses":
        """Deserialize from storage."""
        weights = {k: np.array(v, dtype=np.float32) for k, v in data["weights"].items()}
        return cls(weights=weights, metadata=data.get("metadata", {}))


def learn_one_vs_rest_diagonal_weights(
    X: np.ndarray,
    labels: List[str],
    stakeholders: List[str],
    seed: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LearnedDiagonalLenses:
    """Learn diagonal lens weights using one-vs-rest logistic regression.

    For each stakeholder, we train a binary classifier to distinguish that
    stakeholder's embeddings from all others. The absolute values of the
    learned coefficients indicate which dimensions are most discriminative
    for that stakeholder.

    Args:
        X: Embedding matrix of shape [n_samples, n_dimensions]
        labels: List of stakeholder labels, one per sample
        stakeholders: List of stakeholder names to learn weights for
        seed: Random seed for reproducibility
        C: Regularization strength (smaller = more regularization)
        max_iter: Maximum iterations for logistic regression

    Returns:
        LearnedDiagonalLenses with weights for each stakeholder

    Raises:
        ValueError: If X is not 2D

    Example:
        >>> X = np.random.randn(100, 64)
        >>> labels = ["market"] * 50 + ["mission"] * 50
        >>> learned = learn_one_vs_rest_diagonal_weights(X, labels, ["market", "mission"])
        >>> learned.weights["market"].shape
        (64,)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D [n_samples, n_dims], got shape {X.shape}")

    n, d = X.shape
    y = np.array(labels)

    out: Dict[str, np.ndarray] = {}
    training_stats: Dict[str, Dict[str, Any]] = {}

    for s in stakeholders:
        y_bin = (y == s).astype(int)
        n_positive = y_bin.sum()

        if n_positive == 0:
            # No samples for this stakeholder - use uniform weights
            out[s] = np.ones((d,), dtype=np.float32)
            training_stats[s] = {"n_positive": 0, "n_negative": n, "status": "fallback_uniform"}
            continue

        if n_positive == n:
            # All samples are this stakeholder - use uniform weights
            out[s] = np.ones((d,), dtype=np.float32)
            training_stats[s] = {"n_positive": n, "n_negative": 0, "status": "fallback_uniform"}
            continue

        clf = LogisticRegression(
            penalty="l2",
            C=C,
            solver="liblinear",
            random_state=seed,
            max_iter=max_iter,
        )
        clf.fit(X, y_bin)

        # Extract absolute coefficients as importance weights
        w = np.abs(clf.coef_.reshape(-1))  # [d]

        # Normalize to mean 1.0 to preserve embedding scale
        w_mean = w.mean()
        if w_mean > 1e-8:
            w = w / w_mean
        else:
            w = np.ones_like(w)

        out[s] = w.astype(np.float32)
        training_stats[s] = {
            "n_positive": int(n_positive),
            "n_negative": int(n - n_positive),
            "status": "trained",
            "weight_min": float(w.min()),
            "weight_max": float(w.max()),
            "weight_std": float(w.std()),
        }

    metadata = {
        "seed": seed,
        "C": C,
        "max_iter": max_iter,
        "n_samples": n,
        "n_dims": d,
        "stakeholders": stakeholders,
        "training_stats": training_stats,
    }

    return LearnedDiagonalLenses(weights=out, metadata=metadata)


def learn_from_labeled_corpora(
    corpora: Dict[str, np.ndarray],
    seed: int = 42,
    C: float = 1.0,
) -> LearnedDiagonalLenses:
    """Learn lenses from pre-separated stakeholder corpora.

    This is a convenience function when embeddings are already grouped
    by stakeholder rather than having a flat list with labels.

    Args:
        corpora: Dict mapping stakeholder name to embedding matrix [n_i, d]
        seed: Random seed
        C: Regularization strength

    Returns:
        LearnedDiagonalLenses

    Example:
        >>> corpora = {
        ...     "market": np.random.randn(50, 64),
        ...     "mission": np.random.randn(30, 64),
        ... }
        >>> learned = learn_from_labeled_corpora(corpora)
    """
    # Flatten corpora into X and labels
    X_parts = []
    labels = []

    for stakeholder, embeddings in corpora.items():
        if embeddings.ndim != 2:
            raise ValueError(f"Corpus for {stakeholder} must be 2D, got {embeddings.shape}")
        X_parts.append(embeddings)
        labels.extend([stakeholder] * embeddings.shape[0])

    X = np.vstack(X_parts)
    stakeholders = list(corpora.keys())

    return learn_one_vs_rest_diagonal_weights(X, labels, stakeholders, seed=seed, C=C)
