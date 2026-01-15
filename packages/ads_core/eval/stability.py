"""Stability evaluation under perturbations.

Measures how robust Pareto sets and rankings are to:
1. Embedding noise: small perturbations to embedding vectors
2. Target resampling: bootstrap resampling of target sets
3. Weight perturbation: noise in lens weights

Key metrics:
- Jaccard stability: overlap of Pareto sets across runs
- Rank correlation: Spearman correlation of option rankings
- Objective variance: spread of objective values under perturbation

These stability metrics are important for:
- Validating that recommendations are not artifacts of noise
- Building trust in the recommendation system
- Satisfying reproducibility requirements for publication
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ads_core.eval.pareto import pareto_front, dominates


@dataclass
class StabilityResult:
    """Result of stability evaluation."""

    n_runs: int
    jaccard_mean: float
    jaccard_std: float
    rank_correlation_mean: float
    rank_correlation_std: float
    pareto_frequency: Dict[str, float]  # option_id -> fraction of runs in Pareto
    objective_variance: Dict[str, Dict[str, float]]  # option_id -> {obj: variance}
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_runs": self.n_runs,
            "jaccard_mean": round(self.jaccard_mean, 4),
            "jaccard_std": round(self.jaccard_std, 4),
            "rank_correlation_mean": round(self.rank_correlation_mean, 4),
            "rank_correlation_std": round(self.rank_correlation_std, 4),
            "pareto_frequency": {k: round(v, 4) for k, v in self.pareto_frequency.items()},
            "objective_variance": {
                k: {obj: round(var, 6) for obj, var in v.items()}
                for k, v in self.objective_variance.items()
            },
            "metadata": self.metadata,
        }


@dataclass
class StabilityConfig:
    """Configuration for stability evaluation."""

    n_runs: int = 10
    noise_scale: float = 0.01  # Scale of Gaussian noise
    resample_fraction: float = 0.8  # Fraction of targets to resample
    seed: int = 42


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        set_a: First set
        set_b: Second set

    Returns:
        Jaccard similarity in [0, 1]
    """
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def spearman_rank_correlation(ranks_a: List[int], ranks_b: List[int]) -> float:
    """Compute Spearman rank correlation coefficient.

    Args:
        ranks_a: First ranking
        ranks_b: Second ranking

    Returns:
        Correlation coefficient in [-1, 1]
    """
    n = len(ranks_a)
    if n < 2:
        return 1.0

    # Convert to numpy
    a = np.array(ranks_a)
    b = np.array(ranks_b)

    # Compute rank differences
    d = a - b
    d_squared_sum = np.sum(d ** 2)

    # Spearman formula
    rho = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))
    return float(rho)


def add_gaussian_noise(
    embeddings: Dict[str, np.ndarray],
    scale: float,
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    """Add Gaussian noise to embeddings.

    Args:
        embeddings: Dict of id -> embedding vector
        scale: Standard deviation of noise
        rng: Random state for reproducibility

    Returns:
        Perturbed embeddings
    """
    perturbed = {}
    for k, v in embeddings.items():
        noise = rng.normal(0, scale, size=v.shape).astype(v.dtype)
        perturbed[k] = v + noise
    return perturbed


def resample_targets(
    target_vectors: List[np.ndarray],
    fraction: float,
    rng: np.random.RandomState,
) -> List[np.ndarray]:
    """Bootstrap resample target vectors.

    Args:
        target_vectors: List of target embedding vectors
        fraction: Fraction to resample
        rng: Random state

    Returns:
        Resampled list of vectors
    """
    n = len(target_vectors)
    k = max(1, int(n * fraction))
    indices = rng.choice(n, size=k, replace=True)
    return [target_vectors[i] for i in indices]


class StabilityEvaluator:
    """Evaluator for measuring recommendation stability.

    Runs multiple perturbation experiments and measures:
    - Jaccard stability of Pareto sets
    - Rank correlation of option orderings
    - Variance of objective scores
    """

    def __init__(
        self,
        evaluate_fn: Callable[[Dict[str, np.ndarray]], List[Dict[str, Any]]],
        config: Optional[StabilityConfig] = None,
    ):
        """Initialize stability evaluator.

        Args:
            evaluate_fn: Function that takes embeddings and returns evaluation results
            config: Stability configuration
        """
        self.evaluate_fn = evaluate_fn
        self.config = config or StabilityConfig()

    def evaluate_with_embedding_noise(
        self,
        embeddings: Dict[str, np.ndarray],
        objective_keys: List[str],
    ) -> StabilityResult:
        """Evaluate stability under embedding perturbations.

        Args:
            embeddings: Original embeddings
            objective_keys: Objective names for Pareto computation

        Returns:
            StabilityResult with stability metrics
        """
        rng = np.random.RandomState(self.config.seed)

        # Collect results across runs
        all_pareto_sets: List[set] = []
        all_rankings: List[List[Tuple[str, float]]] = []
        all_objectives: Dict[str, List[Dict[str, float]]] = {}

        for run_idx in range(self.config.n_runs):
            # Perturb embeddings
            perturbed = add_gaussian_noise(embeddings, self.config.noise_scale, rng)

            # Run evaluation
            results = self.evaluate_fn(perturbed)

            # Extract Pareto set
            objectives = [r["objectives"] for r in results]
            pareto_idx = pareto_front(objectives, objective_keys)
            pareto_ids = {results[i]["option_id"] for i in pareto_idx}
            all_pareto_sets.append(pareto_ids)

            # Extract rankings (by sum of objectives)
            rankings = []
            for r in results:
                score = sum(r["objectives"].get(k, 0) for k in objective_keys)
                rankings.append((r["option_id"], score))
            rankings.sort(key=lambda x: -x[1])
            all_rankings.append(rankings)

            # Collect objectives per option
            for r in results:
                oid = r["option_id"]
                if oid not in all_objectives:
                    all_objectives[oid] = []
                all_objectives[oid].append(r["objectives"])

        return self._compute_stability_metrics(
            all_pareto_sets, all_rankings, all_objectives, objective_keys
        )

    def _compute_stability_metrics(
        self,
        pareto_sets: List[set],
        rankings: List[List[Tuple[str, float]]],
        objectives: Dict[str, List[Dict[str, float]]],
        objective_keys: List[str],
    ) -> StabilityResult:
        """Compute stability metrics from collected results."""
        n_runs = len(pareto_sets)

        # Jaccard similarities between consecutive runs
        jaccard_scores = []
        for i in range(n_runs - 1):
            j = jaccard_similarity(pareto_sets[i], pareto_sets[i + 1])
            jaccard_scores.append(j)

        # Also compute Jaccard with first run (reference)
        for i in range(1, n_runs):
            j = jaccard_similarity(pareto_sets[0], pareto_sets[i])
            jaccard_scores.append(j)

        # Rank correlations
        rank_correlations = []
        for i in range(n_runs - 1):
            # Map option IDs to ranks
            rank_a = {oid: rank for rank, (oid, _) in enumerate(rankings[i])}
            rank_b = {oid: rank for rank, (oid, _) in enumerate(rankings[i + 1])}

            common = set(rank_a.keys()) & set(rank_b.keys())
            if len(common) >= 2:
                ranks_a = [rank_a[oid] for oid in common]
                ranks_b = [rank_b[oid] for oid in common]
                rho = spearman_rank_correlation(ranks_a, ranks_b)
                rank_correlations.append(rho)

        # Pareto frequency per option
        all_options = set()
        for ps in pareto_sets:
            all_options.update(ps)

        pareto_frequency = {}
        for oid in all_options:
            count = sum(1 for ps in pareto_sets if oid in ps)
            pareto_frequency[oid] = count / n_runs

        # Objective variance per option
        objective_variance = {}
        for oid, obj_list in objectives.items():
            variances = {}
            for key in objective_keys:
                values = [o.get(key, 0) for o in obj_list]
                variances[key] = float(np.var(values))
            objective_variance[oid] = variances

        return StabilityResult(
            n_runs=n_runs,
            jaccard_mean=float(np.mean(jaccard_scores)) if jaccard_scores else 1.0,
            jaccard_std=float(np.std(jaccard_scores)) if jaccard_scores else 0.0,
            rank_correlation_mean=float(np.mean(rank_correlations)) if rank_correlations else 1.0,
            rank_correlation_std=float(np.std(rank_correlations)) if rank_correlations else 0.0,
            pareto_frequency=pareto_frequency,
            objective_variance=objective_variance,
            metadata={
                "noise_scale": self.config.noise_scale,
                "seed": self.config.seed,
            },
        )


def run_stability_analysis(
    embeddings: Dict[str, np.ndarray],
    target_centroids: Dict[str, np.ndarray],
    lenses: Dict[str, Any],
    objective_keys: List[str],
    constraint_fn: Optional[Callable] = None,
    n_runs: int = 10,
    noise_scale: float = 0.01,
    seed: int = 42,
) -> StabilityResult:
    """Run stability analysis on a recommendation setup.

    Args:
        embeddings: Course embeddings
        target_centroids: Target centroids for each objective
        lenses: Lens transformations
        objective_keys: Objective names
        constraint_fn: Optional constraint function
        n_runs: Number of perturbation runs
        noise_scale: Scale of embedding noise
        seed: Random seed

    Returns:
        StabilityResult with metrics
    """
    from ads_core.eval.objectives import evaluate_option

    def evaluate_fn(perturbed_embeddings: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        results = []
        for oid, emb in perturbed_embeddings.items():
            obj = evaluate_option(emb, target_centroids, lenses)
            result = {"option_id": oid, "objectives": obj, "feasible": True}
            if constraint_fn:
                cons = constraint_fn(obj)
                result["feasible"] = cons.get("feasible", True)
                result["violations"] = cons.get("violations", {})
            results.append(result)
        return results

    config = StabilityConfig(n_runs=n_runs, noise_scale=noise_scale, seed=seed)
    evaluator = StabilityEvaluator(evaluate_fn, config)

    return evaluator.evaluate_with_embedding_noise(embeddings, objective_keys)


def stability_to_csv_rows(result: StabilityResult) -> List[Dict[str, Any]]:
    """Convert stability result to CSV-friendly rows.

    Args:
        result: StabilityResult

    Returns:
        List of row dicts
    """
    rows = []
    for oid, freq in result.pareto_frequency.items():
        row = {
            "option_id": oid,
            "pareto_frequency": freq,
            "stable": freq >= 0.8,  # Threshold for "stable" Pareto membership
        }
        if oid in result.objective_variance:
            for obj, var in result.objective_variance[oid].items():
                row[f"var_{obj}"] = var
        rows.append(row)
    return rows
