"""Item Response Theory models for learner assessment.

IRT separates learner ability (θ) from item characteristics
(difficulty, discrimination), enabling fair assessment across
different tests and calibration of adaptive item selection.

Models:
- 1PL (Rasch): P(correct) = σ(θ - b)
- 2PL: P(correct) = σ(a(θ - b))
- 3PL: P(correct) = c + (1-c)σ(a(θ - b))

where σ = logistic function, θ = ability, a = discrimination,
b = difficulty, c = guessing parameter.

Educational context:
- Items = course assessments, exam questions, project rubrics
- θ = latent competency in a specific dimension
- Discrimination = how well an item separates high/low ability learners
- Fisher information = how much an item tells us about ability → drives adaptive selection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class IRTItem:
    """An assessment item with IRT parameters."""

    item_id: str
    difficulty: float       # b: higher = harder
    discrimination: float   # a: higher = better at separating ability levels
    guessing: float = 0.0   # c: chance of correct answer by guessing (3PL only)
    content_area: str = ""  # which competency this item measures

    def probability(self, theta: float) -> float:
        """P(correct | θ) under 3PL model."""
        z = self.discrimination * (theta - self.difficulty)
        logistic = 1.0 / (1.0 + np.exp(-z))
        return self.guessing + (1.0 - self.guessing) * logistic

    def information(self, theta: float) -> float:
        """Fisher information I(θ) for this item.

        Higher information = the item is more informative at this ability level.
        For 2PL: I(θ) = a² * P(θ) * (1 - P(θ))
        For 3PL: adjusted for guessing.
        """
        p = self.probability(theta)
        # Derivative of P w.r.t. θ
        p_star = 1.0 / (1.0 + np.exp(-self.discrimination * (theta - self.difficulty)))
        if self.guessing > 0:
            # 3PL information
            numerator = self.discrimination ** 2 * (p_star * (1 - p_star)) ** 2
            denominator = p * (1 - p)
            return numerator / max(denominator, 1e-10)
        else:
            # 2PL information (cleaner formula)
            return self.discrimination ** 2 * p * (1 - p)


@dataclass
class IRTResponse:
    """A single response: learner answered an item."""

    learner_id: str
    item_id: str
    correct: bool
    response_time: Optional[float] = None  # seconds


@dataclass
class AbilityEstimate:
    """Estimated ability for a learner."""

    learner_id: str
    theta: float           # point estimate
    se: float              # standard error
    n_items: int           # items administered
    items_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """95% confidence interval."""
        return (self.theta - 1.96 * self.se, self.theta + 1.96 * self.se)


class IRTModel:
    """Item Response Theory model for ability estimation.

    Supports joint estimation of:
    - Learner abilities (θ) via MLE or EAP
    - Item parameters (a, b, c) via marginal MLE (simplified EM)
    """

    def __init__(
        self,
        items: List[IRTItem],
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
    ) -> None:
        self.items = {item.item_id: item for item in items}
        self.prior_mean = prior_mean
        self.prior_std = prior_std

    def estimate_ability_mle(
        self,
        responses: List[IRTResponse],
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> AbilityEstimate:
        """Maximum Likelihood Estimation of θ.

        Uses Newton-Raphson on the log-likelihood.
        """
        if not responses:
            return AbilityEstimate(
                learner_id="", theta=self.prior_mean,
                se=self.prior_std, n_items=0,
            )

        theta = 0.0  # start at average
        learner_id = responses[0].learner_id

        for iteration in range(max_iter):
            # Compute gradient and Hessian of log-likelihood
            grad = 0.0
            hessian = 0.0

            for resp in responses:
                item = self.items.get(resp.item_id)
                if item is None:
                    continue
                p = item.probability(theta)
                p = np.clip(p, 1e-8, 1 - 1e-8)

                u = 1.0 if resp.correct else 0.0
                # For 2PL/3PL, gradient of log P(u|θ)
                p_star = 1.0 / (1.0 + np.exp(-item.discrimination * (theta - item.difficulty)))
                dp_dtheta = item.discrimination * p_star * (1 - p_star)
                if item.guessing > 0:
                    dp_dtheta *= (1 - item.guessing) / max(p, 1e-10)

                grad += (u - p) * dp_dtheta / max(p * (1 - p), 1e-10)
                hessian -= item.information(theta)

            if abs(hessian) < 1e-10:
                break

            delta = -grad / hessian
            theta += np.clip(delta, -1.0, 1.0)  # damped update

            if abs(delta) < tol:
                break

        se = 1.0 / np.sqrt(max(-hessian, 1e-10))
        return AbilityEstimate(
            learner_id=learner_id,
            theta=float(theta),
            se=float(se),
            n_items=len(responses),
            items_used=[r.item_id for r in responses],
        )

    def estimate_ability_eap(
        self,
        responses: List[IRTResponse],
        n_quadrature: int = 41,
    ) -> AbilityEstimate:
        """Expected A Posteriori estimation (Bayesian).

        Integrates over a prior to get the posterior mean.
        More stable than MLE for few responses.
        """
        if not responses:
            return AbilityEstimate(
                learner_id="", theta=self.prior_mean,
                se=self.prior_std, n_items=0,
            )

        learner_id = responses[0].learner_id

        # Quadrature points
        theta_grid = np.linspace(
            self.prior_mean - 4 * self.prior_std,
            self.prior_mean + 4 * self.prior_std,
            n_quadrature,
        )

        # Prior density (normal)
        prior = np.exp(-0.5 * ((theta_grid - self.prior_mean) / self.prior_std) ** 2)
        prior /= prior.sum()

        # Likelihood at each quadrature point
        log_lik = np.zeros(n_quadrature)
        for resp in responses:
            item = self.items.get(resp.item_id)
            if item is None:
                continue
            for k, theta in enumerate(theta_grid):
                p = item.probability(theta)
                p = np.clip(p, 1e-10, 1 - 1e-10)
                if resp.correct:
                    log_lik[k] += np.log(p)
                else:
                    log_lik[k] += np.log(1 - p)

        # Posterior = prior * likelihood (in log space)
        log_posterior = np.log(prior + 1e-300) + log_lik
        log_posterior -= log_posterior.max()  # numerical stability
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum()

        # EAP estimate
        theta_hat = float(np.sum(theta_grid * posterior))
        theta_var = float(np.sum((theta_grid - theta_hat) ** 2 * posterior))
        se = np.sqrt(max(theta_var, 1e-10))

        return AbilityEstimate(
            learner_id=learner_id,
            theta=theta_hat,
            se=se,
            n_items=len(responses),
            items_used=[r.item_id for r in responses],
        )

    def test_information(self, theta: float) -> float:
        """Total test information at ability θ (sum over all items)."""
        return sum(item.information(theta) for item in self.items.values())


def generate_item_bank(
    n_items: int = 50,
    n_areas: int = 3,
    seed: int = 42,
) -> List[IRTItem]:
    """Generate a synthetic item bank for testing.

    Creates items with realistic parameter distributions:
    - difficulty ~ N(0, 1): centered around average ability
    - discrimination ~ LogNormal(0, 0.5): positive, mostly 0.5-2.0
    - guessing = 0 for 2PL
    """
    rng = np.random.RandomState(seed)
    areas = [f"competency_{i}" for i in range(n_areas)]
    items = []

    for i in range(n_items):
        items.append(IRTItem(
            item_id=f"item_{i:03d}",
            difficulty=float(rng.normal(0, 1)),
            discrimination=float(np.exp(rng.normal(0, 0.5))),
            guessing=0.0,
            content_area=areas[i % n_areas],
        ))

    return items
