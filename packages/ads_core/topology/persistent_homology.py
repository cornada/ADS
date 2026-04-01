"""Persistent homology for curriculum embedding spaces.

Topological Data Analysis (TDA) extracts shape features from point clouds
of course embeddings. Key concepts:

- Betti numbers: β₀ = connected components, β₁ = loops, β₂ = voids
- Persistence: features that survive across many scales are "real structure"
- Persistence diagrams: birth/death pairs for topological features

Educational interpretation:
- β₀ (components): distinct curriculum clusters (e.g., separate FGOS domains)
- β₁ (loops): alternative paths to same competency — curriculum redundancy
- Persistent features: fundamental structure of the curriculum space
- Short-lived features: noise in the embedding

Computational notes:
- ripser computes Vietoris-Rips persistence efficiently
- For >1000 points in 384d, UMAP reduction to 10-20d is recommended
- Distance matrix approach for large datasets
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    from ripser import ripser
except ImportError:
    ripser = None  # type: ignore[assignment]

try:
    from persim import plot_diagrams as _plot_diagrams
    from persim import wasserstein as persim_wasserstein
    from persim import bottleneck as persim_bottleneck
except ImportError:
    _plot_diagrams = None  # type: ignore[assignment]
    persim_wasserstein = None  # type: ignore[assignment]
    persim_bottleneck = None  # type: ignore[assignment]


@dataclass
class PersistenceFeature:
    """A single topological feature (birth, death) pair."""

    dimension: int
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        """Lifetime of this feature. Longer = more significant."""
        return self.death - self.birth

    @property
    def midlife(self) -> float:
        return (self.birth + self.death) / 2.0


@dataclass
class TopologicalSummary:
    """Summary of topological analysis on an embedding space."""

    n_points: int
    embedding_dim: int
    reduced_dim: Optional[int]
    max_homology_dim: int
    features: List[PersistenceFeature]
    betti_numbers: Dict[int, int]
    persistence_entropy: Dict[int, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_features(self) -> int:
        return len(self.features)

    def features_by_dim(self, dim: int) -> List[PersistenceFeature]:
        return [f for f in self.features if f.dimension == dim]

    def top_persistent(self, dim: int, k: int = 10) -> List[PersistenceFeature]:
        """Top-k most persistent features in a given dimension."""
        dim_feats = self.features_by_dim(dim)
        return sorted(dim_feats, key=lambda f: f.persistence, reverse=True)[:k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_points": self.n_points,
            "embedding_dim": self.embedding_dim,
            "reduced_dim": self.reduced_dim,
            "max_homology_dim": self.max_homology_dim,
            "betti_numbers": self.betti_numbers,
            "persistence_entropy": {
                k: round(v, 4) for k, v in self.persistence_entropy.items()
            },
            "total_features": self.total_features,
            "features_per_dim": {
                d: len(self.features_by_dim(d))
                for d in range(self.max_homology_dim + 1)
            },
            "top_persistent": {
                d: [
                    {"birth": round(f.birth, 4), "death": round(f.death, 4),
                     "persistence": round(f.persistence, 4)}
                    for f in self.top_persistent(d, k=5)
                ]
                for d in range(self.max_homology_dim + 1)
            },
            "metadata": self.metadata,
        }


# ============================================================================
# Dimensionality Reduction
# ============================================================================

def reduce_embeddings(
    embeddings: np.ndarray,
    target_dim: int = 15,
    method: str = "umap",
    seed: int = 42,
) -> np.ndarray:
    """Reduce high-dimensional embeddings for TDA tractability.

    Args:
        embeddings: (n, d) array of embeddings
        target_dim: Target dimensionality
        method: "umap" or "pca"
        seed: Random seed

    Returns:
        (n, target_dim) reduced embeddings
    """
    n, d = embeddings.shape
    if d <= target_dim:
        return embeddings

    if method == "umap":
        import umap
        reducer = umap.UMAP(
            n_components=target_dim,
            metric="cosine",
            random_state=seed,
            n_neighbors=min(15, n - 1),
            min_dist=0.1,
        )
        return reducer.fit_transform(embeddings)
    elif method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=target_dim, random_state=seed)
        return pca.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'umap' or 'pca'.")


# ============================================================================
# Persistent Homology Computation
# ============================================================================

def compute_persistence(
    embeddings: np.ndarray,
    max_dim: int = 2,
    max_edge_length: Optional[float] = None,
    reduce_dim: Optional[int] = 15,
    reduce_method: str = "pca",
    seed: int = 42,
) -> TopologicalSummary:
    """Compute persistent homology of an embedding point cloud.

    Args:
        embeddings: (n, d) array of course embeddings
        max_dim: Maximum homology dimension (0, 1, or 2)
        max_edge_length: Maximum edge length for Rips complex (None = auto)
        reduce_dim: Reduce to this dim before computing (None = no reduction)
        reduce_method: "umap" or "pca"
        seed: Random seed

    Returns:
        TopologicalSummary with all persistence features
    """
    if ripser is None:
        raise ImportError("ripser is required: pip install ripser")

    n, d = embeddings.shape
    original_dim = d

    # Dimensionality reduction if needed
    reduced_dim = None
    if reduce_dim is not None and d > reduce_dim:
        embeddings = reduce_embeddings(embeddings, reduce_dim, reduce_method, seed)
        reduced_dim = reduce_dim

    # Compute persistence
    kwargs: Dict[str, Any] = {"maxdim": max_dim}
    if max_edge_length is not None:
        kwargs["thresh"] = max_edge_length

    result = ripser(embeddings, **kwargs)
    diagrams = result["dgms"]

    # Extract features
    features: List[PersistenceFeature] = []
    for dim, dgm in enumerate(diagrams):
        for birth, death in dgm:
            if np.isinf(death):
                continue  # skip infinite features for now
            features.append(PersistenceFeature(
                dimension=dim,
                birth=float(birth),
                death=float(death),
            ))

    # Compute Betti numbers (count features with persistence > threshold)
    # Use median persistence as threshold to filter noise
    betti: Dict[int, int] = {}
    entropy: Dict[int, float] = {}
    for dim in range(max_dim + 1):
        dim_feats = [f for f in features if f.dimension == dim]
        if dim_feats:
            persistences = [f.persistence for f in dim_feats]
            threshold = np.median(persistences)
            betti[dim] = sum(1 for p in persistences if p > threshold)
            entropy[dim] = _persistence_entropy(persistences)
        else:
            betti[dim] = 0
            entropy[dim] = 0.0

    return TopologicalSummary(
        n_points=n,
        embedding_dim=original_dim,
        reduced_dim=reduced_dim,
        max_homology_dim=max_dim,
        features=features,
        betti_numbers=betti,
        persistence_entropy=entropy,
        metadata={"reduce_method": reduce_method if reduced_dim else None},
    )


def _persistence_entropy(persistences: List[float]) -> float:
    """Compute persistence entropy (normalized Shannon entropy of lifetimes).

    Higher entropy = more uniform distribution of feature lifetimes.
    Lower entropy = few dominant features + many noise features.
    """
    if len(persistences) <= 1:
        return 0.0
    total = sum(persistences)
    if total == 0:
        return 0.0
    probs = [p / total for p in persistences]
    entropy = -sum(p * np.log(p + 1e-12) for p in probs)
    # Normalize by max possible entropy
    max_entropy = np.log(len(persistences))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


# ============================================================================
# Comparison Between Spaces
# ============================================================================

def compare_topologies(
    summary_a: TopologicalSummary,
    summary_b: TopologicalSummary,
    dim: int = 1,
) -> Dict[str, Any]:
    """Compare topological summaries of two spaces.

    Uses Wasserstein and bottleneck distances between persistence diagrams.

    Args:
        summary_a: First topological summary
        summary_b: Second topological summary
        dim: Homology dimension to compare

    Returns:
        Comparison metrics
    """
    feats_a = summary_a.features_by_dim(dim)
    feats_b = summary_b.features_by_dim(dim)

    dgm_a = np.array([[f.birth, f.death] for f in feats_a]) if feats_a else np.empty((0, 2))
    dgm_b = np.array([[f.birth, f.death] for f in feats_b]) if feats_b else np.empty((0, 2))

    result: Dict[str, Any] = {
        "dimension": dim,
        "n_features_a": len(feats_a),
        "n_features_b": len(feats_b),
        "betti_a": summary_a.betti_numbers.get(dim, 0),
        "betti_b": summary_b.betti_numbers.get(dim, 0),
        "entropy_a": summary_a.persistence_entropy.get(dim, 0.0),
        "entropy_b": summary_b.persistence_entropy.get(dim, 0.0),
    }

    if persim_wasserstein is not None and len(dgm_a) > 0 and len(dgm_b) > 0:
        result["wasserstein_distance"] = float(persim_wasserstein(dgm_a, dgm_b))
    if persim_bottleneck is not None and len(dgm_a) > 0 and len(dgm_b) > 0:
        result["bottleneck_distance"] = float(persim_bottleneck(dgm_a, dgm_b))

    return result


def compare_subsets(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_values: List[str],
    max_dim: int = 1,
    reduce_dim: int = 15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare topology of different subsets (e.g., per-FGOS or per-university).

    Args:
        embeddings: (n, d) full embedding array
        labels: (n,) label for each point
        label_values: Which label values to compare
        max_dim: Maximum homology dimension
        reduce_dim: Dimension reduction target
        seed: Random seed

    Returns:
        Per-label summaries and pairwise comparisons
    """
    summaries = {}
    for label in label_values:
        mask = labels == label
        if mask.sum() < 10:
            continue
        subset = embeddings[mask]
        summaries[label] = compute_persistence(
            subset, max_dim=max_dim, reduce_dim=reduce_dim,
            reduce_method="pca", seed=seed,
        )

    # Pairwise comparisons
    comparisons = {}
    labels_computed = list(summaries.keys())
    for i, la in enumerate(labels_computed):
        for lb in labels_computed[i + 1:]:
            key = f"{la}_vs_{lb}"
            comparisons[key] = compare_topologies(summaries[la], summaries[lb], dim=1)

    return {
        "summaries": {k: v.to_dict() for k, v in summaries.items()},
        "comparisons": comparisons,
    }
