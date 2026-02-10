"""Adaptive item selection for optimal assessment.

Implements Computerized Adaptive Testing (CAT): choose the next
question that maximally reduces uncertainty about the learner's
ability. Instead of giving everyone the same test, each learner
gets items tailored to their estimated level.

Selection criteria:
- Maximum Fisher Information (MFI): choose item with highest I(θ̂)
- Kullback-Leibler (KL): maximize expected KL divergence
- Cost-aware: incorporate measurement cost (time, stress)

The cost-aware criterion is novel to ADS: assessment itself is a
resource in the Pareto balance (time spent testing ≠ time learning).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .irt_model import IRTItem, IRTModel, IRTResponse, AbilityEstimate


@dataclass
class SelectionResult:
    """Result of adaptive item selection."""

    selected_item: IRTItem
    criterion_value: float    # info/KL/cost-adjusted score
    criterion_name: str
    remaining_items: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CATSession:
    """State of an adaptive testing session."""

    learner_id: str
    administered: List[IRTResponse] = field(default_factory=list)
    ability_history: List[AbilityEstimate] = field(default_factory=list)
    terminated: bool = False
    termination_reason: str = ""

    @property
    def n_administered(self) -> int:
        return len(self.administered)

    @property
    def current_estimate(self) -> Optional[AbilityEstimate]:
        return self.ability_history[-1] if self.ability_history else None


class AdaptiveSelector:
    """Adaptive item selector for CAT sessions.

    Workflow:
    1. Initialize session
    2. Loop: select_next → administer → record_response → re-estimate
    3. Check termination (SE threshold or max items)
    """

    def __init__(
        self,
        model: IRTModel,
        item_pool: List[IRTItem],
        criterion: str = "mfi",
        max_items: int = 30,
        se_threshold: float = 0.3,
        item_costs: Optional[Dict[str, float]] = None,
        cost_weight: float = 0.1,
    ) -> None:
        """
        Args:
            model: IRT model for ability estimation
            item_pool: Available items
            criterion: "mfi" (max Fisher info), "kl", or "cost_aware"
            max_items: Maximum items to administer
            se_threshold: Stop when SE < threshold
            item_costs: Cost per item (time/difficulty/stress)
            cost_weight: Weight for cost in cost_aware criterion
        """
        self.model = model
        self.item_pool = list(item_pool)
        self.criterion = criterion
        self.max_items = max_items
        self.se_threshold = se_threshold
        self.item_costs = item_costs or {}
        self.cost_weight = cost_weight

    def start_session(self, learner_id: str) -> CATSession:
        """Initialize a new CAT session."""
        session = CATSession(learner_id=learner_id)
        # Initial estimate from prior
        initial = self.model.estimate_ability_eap([])
        initial = AbilityEstimate(
            learner_id=learner_id,
            theta=initial.theta,
            se=initial.se,
            n_items=0,
        )
        session.ability_history.append(initial)
        return session

    def select_next(self, session: CATSession) -> Optional[SelectionResult]:
        """Select the next item to administer."""
        if session.terminated:
            return None

        # Items not yet administered
        used_ids = {r.item_id for r in session.administered}
        available = [item for item in self.item_pool if item.item_id not in used_ids]

        if not available:
            session.terminated = True
            session.termination_reason = "pool_exhausted"
            return None

        theta_hat = session.current_estimate.theta if session.current_estimate else 0.0

        # Score each available item
        scores = []
        for item in available:
            if self.criterion == "mfi":
                score = item.information(theta_hat)
            elif self.criterion == "kl":
                score = self._kl_criterion(item, theta_hat)
            elif self.criterion == "cost_aware":
                info = item.information(theta_hat)
                cost = self.item_costs.get(item.item_id, 1.0)
                score = info - self.cost_weight * cost
            else:
                score = item.information(theta_hat)

            scores.append(score)

        best_idx = int(np.argmax(scores))
        return SelectionResult(
            selected_item=available[best_idx],
            criterion_value=scores[best_idx],
            criterion_name=self.criterion,
            remaining_items=len(available) - 1,
        )

    def record_response(
        self,
        session: CATSession,
        item_id: str,
        correct: bool,
        response_time: Optional[float] = None,
    ) -> AbilityEstimate:
        """Record a response and update ability estimate."""
        response = IRTResponse(
            learner_id=session.learner_id,
            item_id=item_id,
            correct=correct,
            response_time=response_time,
        )
        session.administered.append(response)

        # Re-estimate ability using EAP (more stable for few items)
        estimate = self.model.estimate_ability_eap(session.administered)
        estimate = AbilityEstimate(
            learner_id=session.learner_id,
            theta=estimate.theta,
            se=estimate.se,
            n_items=session.n_administered,
            items_used=[r.item_id for r in session.administered],
        )
        session.ability_history.append(estimate)

        # Check termination
        if estimate.se < self.se_threshold:
            session.terminated = True
            session.termination_reason = "se_threshold"
        elif session.n_administered >= self.max_items:
            session.terminated = True
            session.termination_reason = "max_items"

        return estimate

    def _kl_criterion(self, item: IRTItem, theta: float, delta: float = 0.1) -> float:
        """KL divergence criterion: how much the posterior changes."""
        p = item.probability(theta)
        p_plus = item.probability(theta + delta)
        p_minus = item.probability(theta - delta)

        kl = 0.0
        for p_ref, p_alt in [(p, p_plus), (p, p_minus)]:
            p_ref = np.clip(p_ref, 1e-10, 1 - 1e-10)
            p_alt = np.clip(p_alt, 1e-10, 1 - 1e-10)
            kl += p_ref * np.log(p_ref / p_alt) + (1 - p_ref) * np.log((1 - p_ref) / (1 - p_alt))

        return kl / 2


def run_cat_simulation(
    model: IRTModel,
    item_pool: List[IRTItem],
    true_theta: float,
    criterion: str = "mfi",
    max_items: int = 30,
    se_threshold: float = 0.3,
    seed: int = 42,
) -> CATSession:
    """Simulate a complete CAT session with known true ability.

    Generates responses probabilistically based on true_theta
    and the item's IRT parameters.
    """
    rng = np.random.RandomState(seed)
    selector = AdaptiveSelector(
        model=model,
        item_pool=item_pool,
        criterion=criterion,
        max_items=max_items,
        se_threshold=se_threshold,
    )

    session = selector.start_session(f"sim_{seed}")

    while not session.terminated:
        result = selector.select_next(session)
        if result is None:
            break

        item = result.selected_item
        p = item.probability(true_theta)
        correct = rng.rand() < p

        selector.record_response(session, item.item_id, correct)

    return session
