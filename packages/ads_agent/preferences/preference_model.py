"""Preference learning from pairwise comparisons.

Instead of hand-crafting reward weights, learn them from expert
preferences: "curriculum A is better than curriculum B because..."

Uses the Bradley-Terry model:
  P(A > B) = σ(r(A) - r(B))

where r(·) is a learned reward function and σ is the logistic function.

Educational applications:
- Faculty preferences: "this course sequence is better than that one"
- Student outcomes: "graduates who took path A had better career outcomes"
- Policy preferences: encode FGOS constraints as soft preferences

This bridges the gap between the Pareto optimization (offline) and
the POMDP controller (online): preferences define the reward function
that the controller optimizes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PreferencePair:
    """A pairwise preference: item_a is preferred over item_b."""

    item_a: str              # ID of preferred item
    item_b: str              # ID of less-preferred item
    features_a: np.ndarray   # feature vector for item A
    features_b: np.ndarray   # feature vector for item B
    strength: float = 1.0    # preference strength (1 = clear, 0.5 = marginal)
    source: str = ""         # who expressed the preference
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearnedReward:
    """A learned reward function from preferences."""

    weights: np.ndarray        # (D,) weight vector
    feature_names: List[str]   # names for each dimension
    n_pairs: int               # number of training pairs
    log_likelihood: float      # final training log-likelihood
    accuracy: float            # training accuracy
    metadata: Dict[str, Any] = field(default_factory=dict)

    def score(self, features: np.ndarray) -> float:
        """Compute reward score for a feature vector."""
        return float(self.weights @ features)

    def rank(self, items: List[np.ndarray]) -> List[int]:
        """Rank items by reward score (descending)."""
        scores = [self.score(f) for f in items]
        return list(np.argsort(scores)[::-1])


class BradleyTerryModel:
    """Bradley-Terry preference model.

    Learns a linear reward function r(x) = w^T x from
    pairwise comparisons using logistic regression.

    P(A > B) = σ(w^T (f_A - f_B))
    """

    def __init__(
        self,
        feature_dim: int,
        regularization: float = 0.01,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.feature_dim = feature_dim
        self.regularization = regularization
        self.feature_names = feature_names or [f"feat_{i}" for i in range(feature_dim)]
        self.weights: Optional[np.ndarray] = None

    def fit(
        self,
        pairs: List[PreferencePair],
        max_iter: int = 100,
        lr: float = 0.1,
        tol: float = 1e-5,
    ) -> LearnedReward:
        """Learn reward weights from preference pairs.

        Uses gradient descent on the logistic loss with L2 regularization.
        """
        n = len(pairs)
        if n == 0:
            self.weights = np.zeros(self.feature_dim)
            return LearnedReward(
                weights=self.weights.copy(),
                feature_names=self.feature_names,
                n_pairs=0, log_likelihood=0.0, accuracy=0.0,
            )

        # Feature differences: f_A - f_B
        diffs = np.array([p.features_a - p.features_b for p in pairs])
        strengths = np.array([p.strength for p in pairs])

        w = np.zeros(self.feature_dim)

        for iteration in range(max_iter):
            # Forward pass
            logits = diffs @ w
            probs = 1.0 / (1.0 + np.exp(-logits))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)

            # Weighted log-likelihood
            ll = float(np.sum(strengths * np.log(probs)))

            # Gradient: ∂LL/∂w = Σ s_i (1 - P_i) * (f_Ai - f_Bi) - λw
            grad = diffs.T @ (strengths * (1 - probs)) - self.regularization * w

            # Update
            w += lr * grad

            if np.linalg.norm(lr * grad) < tol:
                break

        self.weights = w

        # Final metrics
        logits = diffs @ w
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        final_ll = float(np.sum(strengths * np.log(probs)))
        accuracy = float(np.mean(logits > 0))

        return LearnedReward(
            weights=w.copy(),
            feature_names=self.feature_names,
            n_pairs=n,
            log_likelihood=final_ll,
            accuracy=accuracy,
        )

    def predict_preference(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
    ) -> float:
        """P(A preferred over B)."""
        if self.weights is None:
            raise RuntimeError("Must call fit() first")
        logit = float(self.weights @ (features_a - features_b))
        return 1.0 / (1.0 + np.exp(-logit))


def generate_synthetic_preferences(
    n_pairs: int = 100,
    feature_dim: int = 4,
    true_weights: Optional[np.ndarray] = None,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[List[PreferencePair], np.ndarray]:
    """Generate synthetic preference pairs with known ground truth.

    The true reward is r(x) = true_weights^T x + noise.
    A is preferred over B when r(A) > r(B).
    """
    rng = np.random.RandomState(seed)

    if true_weights is None:
        true_weights = rng.randn(feature_dim)
        true_weights /= np.linalg.norm(true_weights)

    pairs = []
    for i in range(n_pairs):
        f_a = rng.randn(feature_dim)
        f_b = rng.randn(feature_dim)

        r_a = true_weights @ f_a + rng.normal(0, noise)
        r_b = true_weights @ f_b + rng.normal(0, noise)

        # Ensure A is preferred (swap if needed)
        if r_a < r_b:
            f_a, f_b = f_b, f_a

        pairs.append(PreferencePair(
            item_a=f"item_{2*i}",
            item_b=f"item_{2*i+1}",
            features_a=f_a,
            features_b=f_b,
            source="synthetic",
        ))

    return pairs, true_weights
