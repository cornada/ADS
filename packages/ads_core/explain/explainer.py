"""Explainability and recourse for multi-objective recommendations.

Explainability:
- For each objective, identify the nearest neighbors in the target corpus
- Explain scores by showing which target artifacts drive the similarity
- Show autonomy drift breakdown for ethics transparency

Recourse:
- Given a weak objective, suggest alternative options that improve it
- Compute trade-offs: what objectives would regress if switching
- Support "bridging" options that improve one objective with minimal loss

Design principles:
- Evidence-based: explanations cite specific artifacts
- Actionable: recourse provides concrete alternatives
- Transparent: all scores and computations are shown
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ads_core.eval.distances import cosine_similarity
from ads_core.eval.pareto import dominates as pareto_dominates
from ads_core.lenses.base import Lens

# Default evidence threshold - can be overridden via constructor
DEFAULT_EVIDENCE_THRESHOLD = 0.5


@dataclass
class Evidence:
    """Evidence for an objective score."""

    artifact_id: str
    text: str
    similarity: float
    contribution: str  # "supports" or "weakens"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "similarity": round(self.similarity, 4),
            "contribution": self.contribution,
        }


@dataclass
class ObjectiveExplanation:
    """Explanation for a single objective score."""

    objective: str
    score: float
    rank: int  # Rank among all candidates for this objective
    percentile: float  # Percentile rank
    evidence: List[Evidence]
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "score": round(self.score, 4),
            "rank": self.rank,
            "percentile": round(self.percentile, 2),
            "evidence": [e.to_dict() for e in self.evidence[:3]],  # Top 3
            "interpretation": self.interpretation,
        }


@dataclass
class DriftExplanation:
    """Explanation of autonomy drift components."""

    total_drift: float
    market_drift: float
    university_drift: float
    learner_score: float
    interpretation: str
    within_threshold: bool
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_drift": round(self.total_drift, 4),
            "market_drift": round(self.market_drift, 4),
            "university_drift": round(self.university_drift, 4),
            "learner_score": round(self.learner_score, 4),
            "interpretation": self.interpretation,
            "within_threshold": self.within_threshold,
            "threshold": self.threshold,
        }


@dataclass
class Explanation:
    """Complete explanation for an option."""

    option_id: str
    option_text: str
    objectives: Dict[str, ObjectiveExplanation]
    drift: Optional[DriftExplanation]
    pareto_status: str  # "pareto", "dominated", "infeasible"
    pareto_reason: str  # Why it is/isn't on Pareto front
    overall_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "option_id": self.option_id,
            "option_text": self.option_text[:200] + "..." if len(self.option_text) > 200 else self.option_text,
            "objectives": {k: v.to_dict() for k, v in self.objectives.items()},
            "drift": self.drift.to_dict() if self.drift else None,
            "pareto_status": self.pareto_status,
            "pareto_reason": self.pareto_reason,
            "overall_summary": self.overall_summary,
        }


@dataclass
class RecourseOption:
    """A suggested alternative option for improving a weak objective."""

    option_id: str
    option_text: str
    target_objective: str  # The objective we're trying to improve
    improvement: float  # How much it improves target
    new_score: float  # New score for target objective
    trade_offs: Dict[str, float]  # Objective -> change (negative = regression)
    net_benefit: float  # Weighted sum of improvements minus regressions
    recommendation: str  # Human-readable recommendation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "option_id": self.option_id,
            "option_text": self.option_text[:200] + "..." if len(self.option_text) > 200 else self.option_text,
            "target_objective": self.target_objective,
            "improvement": round(self.improvement, 4),
            "new_score": round(self.new_score, 4),
            "trade_offs": {k: round(v, 4) for k, v in self.trade_offs.items()},
            "net_benefit": round(self.net_benefit, 4),
            "recommendation": self.recommendation,
        }


@dataclass
class RecourseResult:
    """Recourse analysis result."""

    original_option_id: str
    weak_objective: str
    original_score: float
    alternatives: List[RecourseOption]
    best_alternative: Optional[RecourseOption]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_option_id": self.original_option_id,
            "weak_objective": self.weak_objective,
            "original_score": round(self.original_score, 4),
            "alternatives": [a.to_dict() for a in self.alternatives[:5]],  # Top 5
            "best_alternative": self.best_alternative.to_dict() if self.best_alternative else None,
            "summary": self.summary,
        }


class Explainer:
    """Generates explanations and recourse suggestions for options.

    Provides evidence-based explanations showing which target artifacts
    contribute to objective scores, and suggests alternatives for
    improving weak objectives.
    """

    def __init__(
        self,
        option_embeddings: Dict[str, np.ndarray],
        option_texts: Dict[str, str],
        target_centroids: Dict[str, np.ndarray],
        target_artifacts: Dict[str, List[Tuple[str, str, np.ndarray]]],  # obj -> [(id, text, vec)]
        lenses: Dict[str, Lens],
        all_objectives: Dict[str, Dict[str, float]],  # option_id -> {obj: score}
        pareto_ids: set,
        autonomy_tau: float = 0.25,
        evidence_threshold: Optional[float] = None,
    ):
        """Initialize explainer.

        Args:
            option_embeddings: Dict of option_id -> embedding vector
            option_texts: Dict of option_id -> text
            target_centroids: Dict of objective -> centroid vector
            target_artifacts: Dict of objective -> list of (id, text, vec) tuples
            lenses: Dict of objective -> lens transformation
            all_objectives: Dict of option_id -> objectives dict
            pareto_ids: Set of option IDs on Pareto front
            autonomy_tau: Threshold for autonomy drift constraint
            evidence_threshold: Similarity threshold for evidence contribution
                (default: 0.5, or computed as median if set to None after calibration)
        """
        self.option_embeddings = option_embeddings
        self.option_texts = option_texts
        self.target_centroids = target_centroids
        self.target_artifacts = target_artifacts
        self.lenses = lenses
        self.all_objectives = all_objectives
        self.pareto_ids = pareto_ids
        self.autonomy_tau = autonomy_tau
        self.evidence_threshold = evidence_threshold if evidence_threshold is not None else DEFAULT_EVIDENCE_THRESHOLD

        # Precompute rankings
        self._rankings = self._compute_rankings()

        # Cache objective keys for dominance checks
        self._objective_keys: List[str] = list(target_centroids.keys())

    def _compute_rankings(self) -> Dict[str, List[Tuple[str, float]]]:
        """Compute rankings per objective."""
        rankings = {}
        for obj in self.target_centroids.keys():
            scores = []
            for oid, objs in self.all_objectives.items():
                if obj in objs:
                    scores.append((oid, objs[obj]))
            scores.sort(key=lambda x: -x[1])
            rankings[obj] = scores
        return rankings

    def _get_rank_and_percentile(self, option_id: str, objective: str) -> Tuple[int, float]:
        """Get rank and percentile for an option on an objective."""
        ranking = self._rankings.get(objective, [])
        for i, (oid, _) in enumerate(ranking):
            if oid == option_id:
                rank = i + 1
                percentile = 100 * (1 - i / max(len(ranking), 1))
                return rank, percentile
        return len(ranking), 0.0

    def _find_evidence(
        self,
        option_vec: np.ndarray,
        objective: str,
        top_k: int = 3,
    ) -> List[Evidence]:
        """Find evidence artifacts for an objective score."""
        if objective not in self.target_artifacts:
            return []

        lens = self.lenses.get(objective)
        if lens:
            option_transformed = lens.transform(option_vec)
        else:
            option_transformed = option_vec

        evidence = []
        for art_id, art_text, art_vec in self.target_artifacts[objective]:
            if lens:
                art_transformed = lens.transform(art_vec)
            else:
                art_transformed = art_vec

            sim = cosine_similarity(option_transformed, art_transformed)
            contribution = "supports" if sim > self.evidence_threshold else "weakens"
            evidence.append(Evidence(
                artifact_id=art_id,
                text=art_text,
                similarity=float(sim),
                contribution=contribution,
            ))

        # Sort by similarity
        evidence.sort(key=lambda e: -e.similarity)
        return evidence[:top_k]

    def _interpret_score(self, score: float, objective: str, rank: int, total: int) -> str:
        """Generate interpretation for a score."""
        if score > 0.8:
            strength = "strongly aligned"
        elif score > 0.6:
            strength = "moderately aligned"
        elif score > 0.4:
            strength = "weakly aligned"
        else:
            strength = "poorly aligned"

        position = f"ranked #{rank} of {total}"
        return f"This option is {strength} with {objective} objectives ({position})."

    def _explain_drift(self, objectives: Dict[str, float]) -> Optional[DriftExplanation]:
        """Explain autonomy drift."""
        learner = objectives.get("learner", 0.0)
        market = objectives.get("market", 0.0)
        university = objectives.get("university", 0.0)

        market_drift = max(0.0, market - learner)
        university_drift = max(0.0, university - learner)
        total_drift = 0.5 * market_drift + 0.5 * university_drift

        within = total_drift <= self.autonomy_tau

        if total_drift < 0.1:
            interpretation = "Minimal drift: option aligns primarily with learner preferences."
        elif market_drift > university_drift:
            interpretation = f"Market-driven drift: market alignment ({market:.2f}) exceeds learner ({learner:.2f})."
        elif university_drift > market_drift:
            interpretation = f"University-driven drift: university alignment ({university:.2f}) exceeds learner ({learner:.2f})."
        else:
            interpretation = "Balanced drift from both market and university influences."

        return DriftExplanation(
            total_drift=total_drift,
            market_drift=market_drift,
            university_drift=university_drift,
            learner_score=learner,
            interpretation=interpretation,
            within_threshold=within,
            threshold=self.autonomy_tau,
        )

    def _explain_pareto_status(self, option_id: str, objectives: Dict[str, float]) -> Tuple[str, str]:
        """Explain why option is/isn't on Pareto front.

        Uses the canonical `dominates()` function from pareto module to ensure
        consistency between explanations and Pareto engine output.

        Returns:
            Tuple of (status, reason) where status is 'pareto', 'dominated', or 'infeasible'
        """
        if option_id in self.pareto_ids:
            # Find what objectives it excels at
            best_objs = []
            for obj in objectives:
                rank, _ = self._get_rank_and_percentile(option_id, obj)
                if rank <= 2:
                    best_objs.append(obj)

            if best_objs:
                reason = f"Pareto-optimal: excels at {', '.join(best_objs)} without being dominated on others."
            else:
                reason = "Pareto-optimal: achieves a unique trade-off balance."
            return "pareto", reason

        # Check if infeasible (drift too high)
        drift = self._explain_drift(objectives)
        if drift and not drift.within_threshold:
            return "infeasible", f"Excluded: autonomy drift ({drift.total_drift:.3f}) exceeds threshold ({self.autonomy_tau})."

        # Find dominating option using canonical dominates() function
        # Use only objectives that are present in the current option
        keys = list(objectives.keys())

        for other_id, other_objs in self.all_objectives.items():
            if other_id == option_id:
                continue

            # Check if other dominates this option using the Pareto module
            if pareto_dominates(other_objs, objectives, keys):
                # Find which objectives the dominator is strictly better on
                better_objs = [
                    obj for obj in keys
                    if other_objs.get(obj, 0) > objectives.get(obj, 0)
                ]
                if better_objs:
                    return "dominated", f"Dominated by {other_id} (strictly better on: {', '.join(better_objs)})."
                else:
                    return "dominated", f"Dominated by {other_id} which scores at least as well on all objectives."

        return "dominated", "Dominated by another option with better scores on some objectives."

    def explain(self, option_id: str) -> Explanation:
        """Generate complete explanation for an option.

        Args:
            option_id: ID of option to explain

        Returns:
            Explanation with evidence, drift analysis, and Pareto status
        """
        if option_id not in self.option_embeddings:
            raise ValueError(f"Unknown option: {option_id}")

        option_vec = self.option_embeddings[option_id]
        option_text = self.option_texts.get(option_id, "")
        objectives = self.all_objectives.get(option_id, {})

        # Explain each objective
        obj_explanations = {}
        for obj, score in objectives.items():
            rank, percentile = self._get_rank_and_percentile(option_id, obj)
            total = len(self._rankings.get(obj, []))

            evidence = self._find_evidence(option_vec, obj)
            interpretation = self._interpret_score(score, obj, rank, total)

            obj_explanations[obj] = ObjectiveExplanation(
                objective=obj,
                score=score,
                rank=rank,
                percentile=percentile,
                evidence=evidence,
                interpretation=interpretation,
            )

        # Explain drift
        drift = self._explain_drift(objectives)

        # Explain Pareto status
        pareto_status, pareto_reason = self._explain_pareto_status(option_id, objectives)

        # Generate summary
        best_obj = max(objectives.items(), key=lambda x: x[1]) if objectives else ("none", 0)
        worst_obj = min(objectives.items(), key=lambda x: x[1]) if objectives else ("none", 0)

        summary = (
            f"Option '{option_id}' scores best on {best_obj[0]} ({best_obj[1]:.2f}) "
            f"and weakest on {worst_obj[0]} ({worst_obj[1]:.2f}). "
            f"Status: {pareto_status}."
        )

        return Explanation(
            option_id=option_id,
            option_text=option_text,
            objectives=obj_explanations,
            drift=drift,
            pareto_status=pareto_status,
            pareto_reason=pareto_reason,
            overall_summary=summary,
        )

    def suggest_recourse(
        self,
        option_id: str,
        target_objective: str,
        top_k: int = 3,
        min_improvement: float = 0.05,
    ) -> RecourseResult:
        """Suggest alternatives to improve a weak objective.

        Args:
            option_id: Current option ID
            target_objective: Objective to improve
            top_k: Number of alternatives to return
            min_improvement: Minimum improvement threshold

        Returns:
            RecourseResult with alternative options
        """
        if option_id not in self.all_objectives:
            raise ValueError(f"Unknown option: {option_id}")

        current_objs = self.all_objectives[option_id]
        current_score = current_objs.get(target_objective, 0.0)

        # Find alternatives that improve target objective
        alternatives = []
        for alt_id, alt_objs in self.all_objectives.items():
            if alt_id == option_id:
                continue

            alt_score = alt_objs.get(target_objective, 0.0)
            improvement = alt_score - current_score

            if improvement < min_improvement:
                continue

            # Compute trade-offs
            trade_offs = {}
            for obj in current_objs:
                if obj != target_objective:
                    delta = alt_objs.get(obj, 0) - current_objs.get(obj, 0)
                    trade_offs[obj] = delta

            # Net benefit: improvement minus sum of regressions
            regressions = sum(abs(d) for d in trade_offs.values() if d < 0)
            net_benefit = improvement - 0.5 * regressions

            # Generate recommendation
            if net_benefit > 0.1:
                rec = f"Strong alternative: improves {target_objective} by {improvement:.2f} with minimal trade-offs."
            elif net_benefit > 0:
                rec = f"Moderate alternative: improves {target_objective} by {improvement:.2f} with some trade-offs."
            else:
                regressing = [k for k, v in trade_offs.items() if v < -0.05]
                rec = f"Trade-off: improves {target_objective} but regresses on {', '.join(regressing)}."

            alternatives.append(RecourseOption(
                option_id=alt_id,
                option_text=self.option_texts.get(alt_id, ""),
                target_objective=target_objective,
                improvement=improvement,
                new_score=alt_score,
                trade_offs=trade_offs,
                net_benefit=net_benefit,
                recommendation=rec,
            ))

        # Sort by net benefit
        alternatives.sort(key=lambda a: -a.net_benefit)
        top_alternatives = alternatives[:top_k]

        best = top_alternatives[0] if top_alternatives else None

        if best:
            summary = (
                f"Best recourse for improving {target_objective}: switch to '{best.option_id}' "
                f"for +{best.improvement:.2f} improvement (net benefit: {best.net_benefit:.2f})."
            )
        else:
            summary = f"No alternatives found that improve {target_objective} by at least {min_improvement}."

        return RecourseResult(
            original_option_id=option_id,
            weak_objective=target_objective,
            original_score=current_score,
            alternatives=top_alternatives,
            best_alternative=best,
            summary=summary,
        )


def explain_option(
    option_id: str,
    option_embedding: np.ndarray,
    option_text: str,
    target_centroids: Dict[str, np.ndarray],
    target_artifacts: Dict[str, List[Tuple[str, str, np.ndarray]]],
    lenses: Dict[str, Lens],
    objectives: Dict[str, float],
    pareto_ids: set,
    autonomy_tau: float = 0.25,
) -> Explanation:
    """Convenience function to explain a single option.

    Args:
        option_id: ID of option
        option_embedding: Embedding vector
        option_text: Text content
        target_centroids: Objective centroids
        target_artifacts: Evidence artifacts per objective
        lenses: Lens transformations
        objectives: Objective scores
        pareto_ids: Set of Pareto option IDs
        autonomy_tau: Drift threshold

    Returns:
        Explanation for the option
    """
    explainer = Explainer(
        option_embeddings={option_id: option_embedding},
        option_texts={option_id: option_text},
        target_centroids=target_centroids,
        target_artifacts=target_artifacts,
        lenses=lenses,
        all_objectives={option_id: objectives},
        pareto_ids=pareto_ids,
        autonomy_tau=autonomy_tau,
    )
    return explainer.explain(option_id)


def suggest_recourse(
    option_id: str,
    target_objective: str,
    all_option_embeddings: Dict[str, np.ndarray],
    all_option_texts: Dict[str, str],
    all_objectives: Dict[str, Dict[str, float]],
    target_centroids: Dict[str, np.ndarray],
    lenses: Dict[str, Lens],
    pareto_ids: set,
    top_k: int = 3,
    min_improvement: float = 0.05,
    autonomy_tau: float = 0.25,
) -> RecourseResult:
    """Convenience function to suggest recourse for an option.

    Args:
        option_id: Current option ID
        target_objective: Objective to improve
        all_option_embeddings: All option embeddings
        all_option_texts: All option texts
        all_objectives: All option objective scores
        target_centroids: Objective centroids
        lenses: Lens transformations
        pareto_ids: Set of Pareto option IDs
        top_k: Number of alternatives
        min_improvement: Minimum improvement threshold
        autonomy_tau: Drift threshold

    Returns:
        RecourseResult with alternatives
    """
    explainer = Explainer(
        option_embeddings=all_option_embeddings,
        option_texts=all_option_texts,
        target_centroids=target_centroids,
        target_artifacts={},  # Not needed for recourse
        lenses=lenses,
        all_objectives=all_objectives,
        pareto_ids=pareto_ids,
        autonomy_tau=autonomy_tau,
    )
    return explainer.suggest_recourse(option_id, target_objective, top_k, min_improvement)
