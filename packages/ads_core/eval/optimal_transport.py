"""Optimal Transport analysis for curriculum-market alignment.

Uses Wasserstein distance to measure structural mismatches between
distributions of courses (supply) and job roles (demand) in embedding space.

Key insight vs cosine similarity:
- Cosine similarity: point-to-centroid → collapses demand to single vector
- OT/Wasserstein: distribution-to-distribution → reveals transport plan,
  surplus/deficit zones, and per-element alignment costs

Requires: POT>=0.9 (pip install POT)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TransportResult:
    """Result of an optimal transport computation."""

    wasserstein_distance: float
    transport_plan: np.ndarray  # [n_source, n_target] coupling matrix
    source_labels: List[str]
    target_labels: List[str]
    cost_matrix: np.ndarray  # [n_source, n_target] pairwise costs
    source_weights: np.ndarray
    target_weights: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_source(self) -> int:
        return len(self.source_labels)

    @property
    def n_target(self) -> int:
        return len(self.target_labels)


@dataclass
class ResidualAnalysis:
    """Analysis of transport residuals — what's surplus and what's unmet."""

    surplus_indices: List[int]  # source items with mass left over
    surplus_labels: List[str]
    surplus_scores: List[float]  # how much mass is "wasted"

    deficit_indices: List[int]  # target items with unmet demand
    deficit_labels: List[str]
    deficit_scores: List[float]  # how much demand is unmet

    per_source_cost: np.ndarray  # avg transport cost per source item
    per_target_coverage: np.ndarray  # fraction of target demand satisfied


@dataclass
class OTComparison:
    """Comparison of two distributions via OT."""

    name_a: str
    name_b: str
    wasserstein: float
    wasserstein_normalized: float  # normalized by median pairwise cost
    surplus_ratio: float  # fraction of supply with low utilization
    deficit_ratio: float  # fraction of demand poorly covered
    top_flows: List[Dict[str, Any]]  # strongest source→target connections


def cosine_cost_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute cosine distance cost matrix between two sets of embeddings.

    Args:
        X: Source embeddings [n, d]
        Y: Target embeddings [m, d]

    Returns:
        Cost matrix [n, m] with values in [0, 2]
    """
    # Normalize rows
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    # Cosine distance = 1 - cosine_similarity
    sim = X_norm @ Y_norm.T
    return 1.0 - sim


def compute_transport(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    source_labels: List[str],
    target_labels: List[str],
    source_weights: Optional[np.ndarray] = None,
    target_weights: Optional[np.ndarray] = None,
    regularization: float = 0.05,
    metric: str = "cosine",
) -> TransportResult:
    """Compute optimal transport plan between source and target distributions.

    Args:
        source_embeddings: [n, d] source point embeddings (e.g., courses)
        target_embeddings: [m, d] target point embeddings (e.g., jobs)
        source_labels: names for source points
        target_labels: names for target points
        source_weights: mass per source point (default: uniform)
        target_weights: mass per target point (default: uniform)
        regularization: entropic regularization strength (0 = exact, >0 = Sinkhorn)
        metric: "cosine" or "euclidean"

    Returns:
        TransportResult with coupling matrix and distances
    """
    import ot

    n, m = len(source_embeddings), len(target_embeddings)

    # Build cost matrix
    if metric == "cosine":
        C = cosine_cost_matrix(source_embeddings, target_embeddings)
    else:
        C = ot.dist(source_embeddings, target_embeddings, metric="euclidean")
    C = C.astype(np.float64)

    # Weights (uniform by default, normalized to sum to 1)
    if source_weights is None:
        a = np.ones(n, dtype=np.float64) / n
    else:
        a = source_weights.astype(np.float64)
        a = a / a.sum()

    if target_weights is None:
        b = np.ones(m, dtype=np.float64) / m
    else:
        b = target_weights.astype(np.float64)
        b = b / b.sum()

    # Compute OT plan
    if regularization > 0:
        T = ot.sinkhorn(a, b, C, reg=regularization, numItermax=2000)
    else:
        T = ot.emd(a, b, C)

    # Wasserstein distance = <C, T>
    wasserstein = float(np.sum(C * T))

    return TransportResult(
        wasserstein_distance=wasserstein,
        transport_plan=T,
        source_labels=source_labels,
        target_labels=target_labels,
        cost_matrix=C,
        source_weights=a,
        target_weights=b,
        metadata={
            "regularization": regularization,
            "metric": metric,
            "n_source": n,
            "n_target": m,
        },
    )


def analyze_residuals(
    result: TransportResult,
    surplus_threshold: float = 0.3,
    deficit_threshold: float = 0.3,
) -> ResidualAnalysis:
    """Analyze transport residuals to find surplus supply and unmet demand.

    A source item is "surplus" if most of its mass flows to distant targets
    (high transport cost). A target item has "deficit" if little mass reaches it.

    Args:
        result: TransportResult from compute_transport
        surplus_threshold: fraction of sources to flag as surplus
        deficit_threshold: fraction of targets to flag as having deficit

    Returns:
        ResidualAnalysis with surplus/deficit identification
    """
    T = result.transport_plan
    C = result.cost_matrix

    # Per-source: weighted average transport cost
    # (how "far" does each source item need to ship its mass?)
    source_mass = T.sum(axis=1)
    per_source_cost = np.zeros(result.n_source)
    for i in range(result.n_source):
        if source_mass[i] > 1e-12:
            per_source_cost[i] = (T[i] * C[i]).sum() / source_mass[i]
        else:
            per_source_cost[i] = C[i].mean()  # no mass → use average cost

    # Per-target: fraction of demand satisfied (mass received / weight)
    target_received = T.sum(axis=0)
    per_target_coverage = target_received / (result.target_weights + 1e-12)

    # Identify surplus sources (highest transport cost)
    n_surplus = max(1, int(surplus_threshold * result.n_source))
    surplus_idx = np.argsort(-per_source_cost)[:n_surplus].tolist()

    # Identify deficit targets (lowest coverage)
    n_deficit = max(1, int(deficit_threshold * result.n_target))
    deficit_idx = np.argsort(per_target_coverage)[:n_deficit].tolist()

    return ResidualAnalysis(
        surplus_indices=surplus_idx,
        surplus_labels=[result.source_labels[i] for i in surplus_idx],
        surplus_scores=[float(per_source_cost[i]) for i in surplus_idx],
        deficit_indices=deficit_idx,
        deficit_labels=[result.target_labels[i] for i in deficit_idx],
        deficit_scores=[float(per_target_coverage[i]) for i in deficit_idx],
        per_source_cost=per_source_cost,
        per_target_coverage=per_target_coverage,
    )


def extract_top_flows(
    result: TransportResult,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """Extract the strongest source→target flows from the transport plan.

    Args:
        result: TransportResult from compute_transport
        top_k: number of top flows to return

    Returns:
        List of dicts with source, target, flow_mass, cost
    """
    T = result.transport_plan
    C = result.cost_matrix

    # Flatten and get top-k
    flat_indices = np.argsort(-T.ravel())[:top_k]
    rows, cols = np.unravel_index(flat_indices, T.shape)

    flows = []
    for r, c in zip(rows, cols):
        flow_mass = float(T[r, c])
        if flow_mass < 1e-12:
            continue
        flows.append({
            "source_idx": int(r),
            "target_idx": int(c),
            "source": result.source_labels[r],
            "target": result.target_labels[c],
            "flow_mass": round(flow_mass, 6),
            "cost": round(float(C[r, c]), 4),
        })

    return flows


def compare_distributions(
    name_a: str,
    name_b: str,
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    labels_a: List[str],
    labels_b: List[str],
    regularization: float = 0.05,
) -> OTComparison:
    """Full OT comparison of two distributions with summary metrics.

    Args:
        name_a: Name of first distribution (e.g., "MISIS_courses")
        name_b: Name of second distribution (e.g., "HeadHunter_jobs")
        embeddings_a: [n, d] embeddings for distribution A
        embeddings_b: [m, d] embeddings for distribution B
        labels_a: labels for distribution A
        labels_b: labels for distribution B
        regularization: Sinkhorn regularization

    Returns:
        OTComparison with all summary statistics
    """
    result = compute_transport(
        embeddings_a, embeddings_b,
        labels_a, labels_b,
        regularization=regularization,
    )
    residuals = analyze_residuals(result)
    top_flows = extract_top_flows(result, top_k=20)

    # Normalized Wasserstein: divide by median pairwise cost
    median_cost = float(np.median(result.cost_matrix))
    wasserstein_norm = result.wasserstein_distance / max(median_cost, 1e-8)

    # Surplus ratio: fraction of sources with above-median transport cost
    median_source_cost = float(np.median(residuals.per_source_cost))
    surplus_ratio = float(np.mean(residuals.per_source_cost > median_source_cost * 1.5))

    # Deficit ratio: fraction of targets with below-median coverage
    median_coverage = float(np.median(residuals.per_target_coverage))
    deficit_ratio = float(np.mean(residuals.per_target_coverage < median_coverage * 0.5))

    return OTComparison(
        name_a=name_a,
        name_b=name_b,
        wasserstein=result.wasserstein_distance,
        wasserstein_normalized=wasserstein_norm,
        surplus_ratio=surplus_ratio,
        deficit_ratio=deficit_ratio,
        top_flows=top_flows,
    )
