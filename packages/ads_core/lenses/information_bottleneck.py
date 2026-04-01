"""Information Bottleneck analysis for lens evaluation.

The Information Bottleneck (IB) framework formalizes the lens transformation
as information compression: a good lens compresses the full embedding (X)
while preserving predictive information about the downstream task (Y).

IB objective: min I(X;T) - β * I(T;Y)
  where T = lens(X) is the compressed representation

In ADS context:
- X = full embedding (384d)
- T = lens-transformed embedding (weighted dimensions)
- Y = downstream proxy (FGOS classification, market alignment score, etc.)
- β = trade-off parameter: higher β → preserve more prediction

Key metrics:
- MI(X;T): mutual information between original and compressed (compression)
- MI(T;Y): mutual information between compressed and labels (prediction)
- IB curve: parametric curve in (compression, prediction) plane
- Optimal lens dimension: elbow point on the IB curve

Estimation uses k-NN mutual information estimator (KSG) for continuous
variables and discrete MI for categorical labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class IBResult:
    """Result of Information Bottleneck analysis."""

    compression: float  # I(X;T) — how much is compressed
    prediction: float   # I(T;Y) — how much prediction is preserved
    beta: float         # trade-off parameter
    n_dims: int         # dimension of compressed representation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IBCurve:
    """IB curve: set of (compression, prediction) points."""

    points: List[IBResult]
    optimal_dim: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_points": len(self.points),
            "optimal_dim": self.optimal_dim,
            "curve": [
                {
                    "n_dims": p.n_dims,
                    "compression": round(p.compression, 4),
                    "prediction": round(p.prediction, 4),
                }
                for p in self.points
            ],
            "metadata": self.metadata,
        }


# ============================================================================
# Mutual Information Estimation
# ============================================================================

def _knn_mi(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    """Estimate mutual information I(X;Y) using k-NN (KSG estimator).

    Uses the Kraskov-Stögbauer-Grassberger estimator, which is
    consistent and adaptive to the local density of points.

    Args:
        X: (n, dx) array
        Y: (n, dy) array
        k: Number of nearest neighbors

    Returns:
        Estimated MI in nats
    """
    from scipy.special import digamma
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    if n < k + 1:
        return 0.0

    # Joint space
    XY = np.hstack([X, Y])

    # Find k-th neighbor distance in joint space (Chebyshev/max norm)
    nn_joint = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev")
    nn_joint.fit(XY)
    distances, _ = nn_joint.kneighbors(XY)
    eps = distances[:, k]  # k-th neighbor distance for each point

    # Count neighbors within eps in marginal spaces
    nn_x = NearestNeighbors(metric="chebyshev")
    nn_x.fit(X)
    nn_y = NearestNeighbors(metric="chebyshev")
    nn_y.fit(Y)

    nx = np.zeros(n)
    ny = np.zeros(n)
    for i in range(n):
        # Count points within eps[i] in X-space (excluding self)
        nx[i] = max(nn_x.radius_neighbors([X[i]], radius=eps[i],
                     return_distance=False)[0].shape[0] - 1, 0)
        ny[i] = max(nn_y.radius_neighbors([Y[i]], radius=eps[i],
                     return_distance=False)[0].shape[0] - 1, 0)

    # KSG estimator
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return max(float(mi), 0.0)


def _discrete_mi(X: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Estimate MI I(X; Y_discrete) where Y is categorical.

    Uses entropy difference: I(X;Y) = H(X) - H(X|Y)
    Approximated via k-NN entropy estimator on X, conditioned on Y classes.

    Args:
        X: (n, d) continuous array
        labels: (n,) categorical labels
        k: k for k-NN entropy estimation

    Returns:
        Estimated MI in nats
    """
    from scipy.special import digamma

    n, d = X.shape
    unique_labels = np.unique(labels)

    # H(X) via k-NN (Kozachenko-Leonenko estimator)
    h_x = _kl_entropy(X, k)

    # H(X|Y) = sum_y P(Y=y) * H(X|Y=y)
    h_x_given_y = 0.0
    for label in unique_labels:
        mask = labels == label
        n_y = mask.sum()
        if n_y <= k:
            continue
        p_y = n_y / n
        h_x_given_y += p_y * _kl_entropy(X[mask], k)

    mi = h_x - h_x_given_y
    return max(float(mi), 0.0)


def _kl_entropy(X: np.ndarray, k: int = 5) -> float:
    """Kozachenko-Leonenko k-NN entropy estimator.

    H(X) ≈ (d/n) Σ log(ε_i) + log(n) + C_E + d * log(2) - digamma(k)
    where ε_i = distance to k-th neighbor of point i.
    """
    from scipy.special import digamma
    from sklearn.neighbors import NearestNeighbors

    n, d = X.shape
    if n <= k:
        return 0.0

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    eps = distances[:, k]

    # Avoid log(0)
    eps = np.maximum(eps, 1e-12)

    # Volume of unit ball in d dimensions
    from scipy.special import gammaln
    log_vol = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1)

    h = d * np.mean(np.log(eps)) + np.log(n) + log_vol - digamma(k)
    return float(h)


# ============================================================================
# IB Analysis
# ============================================================================

def compute_ib_point(
    full_embeddings: np.ndarray,
    compressed_embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
) -> IBResult:
    """Compute one point on the IB curve.

    Args:
        full_embeddings: (n, d_full) original embeddings
        compressed_embeddings: (n, d_compressed) lens-transformed
        labels: (n,) categorical labels for downstream task
        k: k for MI estimation

    Returns:
        IBResult with compression and prediction scores
    """
    # Compression: how different is T from X?
    # Use variance ratio as a proxy for I(X;T)
    var_full = np.var(full_embeddings, axis=0).sum()
    var_compressed = np.var(compressed_embeddings, axis=0).sum()
    compression = 1.0 - (var_compressed / max(var_full, 1e-12))

    # Prediction: I(T; Y) — how well can we predict Y from T?
    prediction = _discrete_mi(compressed_embeddings, labels, k=k)

    return IBResult(
        compression=compression,
        prediction=prediction,
        beta=0.0,
        n_dims=compressed_embeddings.shape[1],
    )


def compute_ib_curve(
    embeddings: np.ndarray,
    labels: np.ndarray,
    dims: Optional[List[int]] = None,
    method: str = "pca",
    k: int = 5,
    seed: int = 42,
) -> IBCurve:
    """Compute IB curve by varying compression level (dimensionality).

    Sweeps across different PCA dimensions to trace the compression-prediction
    tradeoff curve. The optimal dimension is at the "elbow" where adding more
    dimensions yields diminishing returns in prediction.

    Args:
        embeddings: (n, d) original embeddings
        labels: (n,) categorical labels
        dims: List of dimensions to try (default: log-spaced)
        method: "pca" for PCA-based compression
        k: k for MI estimation
        seed: Random seed

    Returns:
        IBCurve with points and optimal dimension
    """
    from sklearn.decomposition import PCA

    n, d = embeddings.shape

    if dims is None:
        # Log-spaced dimensions from 2 to d
        max_dim = min(d, n - 1)
        dims = sorted(set([2, 5, 10, 20, 50, 100, 200, max_dim]))
        dims = [dim for dim in dims if dim <= max_dim]

    points: List[IBResult] = []

    for dim in dims:
        if dim >= d or dim >= n:
            continue

        # Compress via PCA
        pca = PCA(n_components=dim, random_state=seed)
        compressed = pca.fit_transform(embeddings)

        # Compute IB point
        point = compute_ib_point(embeddings, compressed, labels, k=k)
        point.n_dims = dim
        point.metadata["explained_variance"] = float(sum(pca.explained_variance_ratio_))
        points.append(point)

    # Find optimal dimension (elbow detection via max curvature)
    optimal_dim = _find_elbow(points)

    return IBCurve(
        points=points,
        optimal_dim=optimal_dim,
        metadata={"method": method, "k": k, "n_samples": n, "original_dim": d},
    )


def _find_elbow(points: List[IBResult]) -> Optional[int]:
    """Find elbow point on the IB curve (max curvature).

    Returns the dimension at the elbow, or None if no clear elbow.
    """
    if len(points) < 3:
        return points[-1].n_dims if points else None

    # Use the prediction values
    predictions = [p.prediction for p in points]
    dims = [p.n_dims for p in points]

    # Normalize
    pred_range = max(predictions) - min(predictions)
    dim_range = max(dims) - min(dims)
    if pred_range == 0 or dim_range == 0:
        return dims[-1]

    norm_pred = [(p - min(predictions)) / pred_range for p in predictions]
    norm_dim = [(d - min(dims)) / dim_range for d in dims]

    # Distance from each point to the line connecting first and last point
    x0, y0 = norm_dim[0], norm_pred[0]
    x1, y1 = norm_dim[-1], norm_pred[-1]

    max_dist = 0.0
    best_idx = 0
    for i in range(1, len(points) - 1):
        xi, yi = norm_dim[i], norm_pred[i]
        # Point-to-line distance
        num = abs((y1 - y0) * xi - (x1 - x0) * yi + x1 * y0 - y1 * x0)
        den = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
        dist = num / max(den, 1e-12)
        if dist > max_dist:
            max_dist = dist
            best_idx = i

    return dims[best_idx]


def evaluate_lens_ib(
    embeddings: np.ndarray,
    lens_weights: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
) -> IBResult:
    """Evaluate a specific diagonal lens using IB metrics.

    Args:
        embeddings: (n, d) original embeddings
        lens_weights: (d,) diagonal lens weights
        labels: (n,) categorical labels
        k: k for MI estimation

    Returns:
        IBResult for this lens
    """
    compressed = embeddings * lens_weights[np.newaxis, :]
    return compute_ib_point(embeddings, compressed, labels, k=k)
