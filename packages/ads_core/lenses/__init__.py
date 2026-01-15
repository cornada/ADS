"""Stakeholder lenses for multi-perspective embedding transformations.

Lenses are the core ADS novelty: L_a(v) = W_a * v

Each stakeholder can have a different lens that emphasizes different dimensions
of the embedding space. This allows the same course embedding to be evaluated
differently from each stakeholder's perspective.

Available Lens Types:
- IdentityLens: L(v) = v (baseline, no transformation)
- DiagonalLens: L(v) = diag(w) * v (element-wise scaling)
- LearnedDiagonalLens: Weights learned via one-vs-rest classification

Example:
    from ads_core.lenses import IdentityLens, DiagonalLens, create_uniform_lens

    # Identity lens (baseline)
    identity = IdentityLens("identity:market")

    # Uniform diagonal lens
    uniform = create_uniform_lens("diagonal:market", dim=768)

    # Custom weighted lens
    weights = np.array([1.0, 2.0, 0.5, ...])
    custom = DiagonalLens("custom:market", weights)

    # Transform an embedding
    v_transformed = custom.transform(v)
"""
from ads_core.lenses.base import Lens
from ads_core.lenses.identity import IdentityLens
from ads_core.lenses.diagonal import (
    DiagonalLens,
    create_uniform_lens,
    create_random_lens,
)
from ads_core.lenses.learn_lenses import (
    LearnedDiagonalLenses,
    learn_one_vs_rest_diagonal_weights,
    learn_from_labeled_corpora,
)

__all__ = [
    # Base
    "Lens",
    # Identity
    "IdentityLens",
    # Diagonal
    "DiagonalLens",
    "create_uniform_lens",
    "create_random_lens",
    # Learned
    "LearnedDiagonalLenses",
    "learn_one_vs_rest_diagonal_weights",
    "learn_from_labeled_corpora",
]
