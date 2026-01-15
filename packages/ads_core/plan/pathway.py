"""Pathway planning with beam search for multi-step course sequences.

A pathway is an ordered sequence of courses (e.g., [course:intro, course:dl, course:ethics])
that a learner might take. The planner evaluates multi-step pathways rather than
individual courses, enabling reasoning about curriculum sequences.

Key concepts:
- Pathway: ordered list of course IDs
- Depth: number of courses in the pathway (e.g., depth=2 means pairs)
- Beam search: keep top-k partial pathways at each depth level
- Pathway objectives: aggregated from individual course objectives

Aggregation strategies:
- mean: average objectives across all courses in pathway
- final: only use the final course's objectives
- cumulative: weighted sum favoring later courses
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ads_core.eval.distances import cosine_similarity
from ads_core.eval.pareto import pareto_front, ParetoResult


@dataclass
class Pathway:
    """A sequence of courses forming a learning pathway."""

    course_ids: List[str]
    objectives: Dict[str, float]
    feasible: bool = True
    violations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pathway_id(self) -> str:
        """Unique identifier for this pathway."""
        return " -> ".join(self.course_ids)

    @property
    def depth(self) -> int:
        """Number of courses in the pathway."""
        return len(self.course_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pathway_id": self.pathway_id,
            "course_ids": self.course_ids,
            "depth": self.depth,
            "objectives": {k: round(v, 6) for k, v in self.objectives.items()},
            "feasible": self.feasible,
            "violations": self.violations,
            "metadata": self.metadata,
        }


@dataclass
class PathwayPlannerConfig:
    """Configuration for pathway planning."""

    max_depth: int = 3
    beam_width: int = 10
    aggregation: str = "mean"  # "mean", "final", "cumulative"
    allow_repeats: bool = False
    objective_keys: Optional[List[str]] = None


class PathwayPlanner:
    """Beam search planner for multi-step course pathways.

    The planner explores sequences of courses using beam search:
    1. Start with all individual courses as depth-1 pathways
    2. At each depth, extend top-k pathways with all valid courses
    3. Evaluate extended pathways and keep top-k by Pareto rank + crowding

    Example:
        planner = PathwayPlanner(course_embeddings, evaluator, config)
        pathways = planner.plan(max_depth=3, beam_width=10)
    """

    def __init__(
        self,
        course_embeddings: Dict[str, np.ndarray],
        evaluate_fn: Callable[[np.ndarray], Dict[str, float]],
        config: Optional[PathwayPlannerConfig] = None,
    ):
        """Initialize pathway planner.

        Args:
            course_embeddings: Dict mapping course_id to embedding vector
            evaluate_fn: Function that takes an embedding and returns objective scores
            config: Planning configuration
        """
        self.course_embeddings = course_embeddings
        self.evaluate_fn = evaluate_fn
        self.config = config or PathwayPlannerConfig()
        self.course_ids = list(course_embeddings.keys())

    def _aggregate_pathway_embedding(
        self,
        pathway: List[str],
        method: str = "mean",
    ) -> np.ndarray:
        """Aggregate embeddings for a pathway.

        Args:
            pathway: List of course IDs
            method: Aggregation method ("mean", "final", "cumulative")

        Returns:
            Aggregated embedding vector
        """
        embeddings = [self.course_embeddings[cid] for cid in pathway]

        if method == "final":
            return embeddings[-1]
        elif method == "cumulative":
            # Weighted sum: later courses have higher weight
            weights = np.arange(1, len(embeddings) + 1, dtype=np.float32)
            weights = weights / weights.sum()
            stacked = np.stack(embeddings, axis=0)
            return (stacked * weights[:, None]).sum(axis=0)
        else:  # mean
            return np.mean(embeddings, axis=0)

    def _evaluate_pathway(self, pathway: List[str]) -> Pathway:
        """Evaluate a pathway's objectives.

        Args:
            pathway: List of course IDs

        Returns:
            Pathway with objectives computed
        """
        agg_embedding = self._aggregate_pathway_embedding(
            pathway, self.config.aggregation
        )
        objectives = self.evaluate_fn(agg_embedding)

        return Pathway(
            course_ids=pathway,
            objectives=objectives,
            metadata={"aggregation": self.config.aggregation},
        )

    def _extend_pathway(self, pathway: List[str]) -> List[List[str]]:
        """Generate all valid extensions of a pathway.

        Args:
            pathway: Current pathway

        Returns:
            List of extended pathways
        """
        extensions = []
        for course_id in self.course_ids:
            if not self.config.allow_repeats and course_id in pathway:
                continue
            extensions.append(pathway + [course_id])
        return extensions

    def _select_beam(
        self,
        pathways: List[Pathway],
        beam_width: int,
        keys: Sequence[str],
    ) -> List[Pathway]:
        """Select top pathways for beam using Pareto rank + crowding.

        Args:
            pathways: Candidate pathways
            beam_width: Number to keep
            keys: Objective names

        Returns:
            Selected pathways
        """
        if len(pathways) <= beam_width:
            return pathways

        # Compute Pareto ranks
        objectives = [p.objectives for p in pathways]
        from ads_core.eval.pareto import pareto_rank
        ranks = pareto_rank(objectives, keys)

        # Sort by rank, then by crowding distance (diversity)
        # For simplicity, use sum of objectives as tiebreaker
        scored = []
        for i, (p, rank) in enumerate(zip(pathways, ranks)):
            obj_sum = sum(p.objectives.values())
            scored.append((rank, -obj_sum, i, p))

        scored.sort(key=lambda x: (x[0], x[1]))
        return [s[3] for s in scored[:beam_width]]

    def plan(
        self,
        max_depth: Optional[int] = None,
        beam_width: Optional[int] = None,
        apply_constraints: Optional[Callable[[Dict[str, float]], Dict[str, Any]]] = None,
    ) -> List[Pathway]:
        """Run beam search to find good pathways.

        Args:
            max_depth: Maximum pathway length (default from config)
            beam_width: Beam width (default from config)
            apply_constraints: Optional function to check feasibility

        Returns:
            List of Pareto-optimal pathways at max depth
        """
        max_depth = max_depth or self.config.max_depth
        beam_width = beam_width or self.config.beam_width
        keys = self.config.objective_keys or ["market", "mission", "university", "learner"]

        # Initialize with depth-1 pathways (individual courses)
        current_beam: List[Pathway] = []
        for course_id in self.course_ids:
            pathway = self._evaluate_pathway([course_id])
            if apply_constraints:
                cons = apply_constraints(pathway.objectives)
                pathway.feasible = cons.get("feasible", True)
                pathway.violations = cons.get("violations", {})
            current_beam.append(pathway)

        # Beam search for deeper pathways
        for depth in range(2, max_depth + 1):
            candidates: List[Pathway] = []

            for pathway in current_beam:
                extensions = self._extend_pathway(pathway.course_ids)
                for ext in extensions:
                    new_pathway = self._evaluate_pathway(ext)
                    if apply_constraints:
                        cons = apply_constraints(new_pathway.objectives)
                        new_pathway.feasible = cons.get("feasible", True)
                        new_pathway.violations = cons.get("violations", {})
                    candidates.append(new_pathway)

            # Select best pathways for next iteration
            current_beam = self._select_beam(candidates, beam_width, keys)

        # Return Pareto-optimal pathways from final beam
        objectives = [p.objectives for p in current_beam]
        pareto_idx = pareto_front(objectives, keys)
        return [current_beam[i] for i in pareto_idx]


def plan_pathways(
    course_embeddings: Dict[str, np.ndarray],
    target_centroids: Dict[str, np.ndarray],
    lenses: Dict[str, Any],
    max_depth: int = 2,
    beam_width: int = 10,
    aggregation: str = "mean",
    constraint_fn: Optional[Callable] = None,
) -> List[Pathway]:
    """Convenience function to plan pathways.

    Args:
        course_embeddings: Dict of course_id -> embedding
        target_centroids: Dict of objective_name -> centroid
        lenses: Dict of objective_name -> lens transformation
        max_depth: Maximum pathway depth
        beam_width: Beam search width
        aggregation: How to aggregate pathway embeddings
        constraint_fn: Optional constraint checker

    Returns:
        List of Pareto-optimal pathways
    """
    from ads_core.eval.objectives import evaluate_option

    def evaluate_fn(embedding: np.ndarray) -> Dict[str, float]:
        return evaluate_option(embedding, target_centroids, lenses)

    config = PathwayPlannerConfig(
        max_depth=max_depth,
        beam_width=beam_width,
        aggregation=aggregation,
        objective_keys=list(target_centroids.keys()),
    )

    planner = PathwayPlanner(course_embeddings, evaluate_fn, config)
    return planner.plan(apply_constraints=constraint_fn)


def pathways_to_json(pathways: List[Pathway]) -> List[Dict[str, Any]]:
    """Convert pathways to JSON-serializable format."""
    return [p.to_dict() for p in pathways]
