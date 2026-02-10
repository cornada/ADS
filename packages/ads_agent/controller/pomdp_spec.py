"""POMDP specification for curriculum recommendation.

Models the tutoring interaction as a Partially Observable Markov
Decision Process: the agent doesn't know the student's true competency
(hidden state), only observes noisy signals (test scores, engagement).

Key insight: the agent must balance:
- Exploration: give diagnostic items to reduce uncertainty
- Exploitation: recommend courses the student is likely to benefit from
- Risk management: avoid overwhelming/boring recommendations

POMDP components:
- States: learner competency vector (continuous, discretized for solver)
- Actions: course recommendations, assessments, rest
- Observations: test scores, completion times, engagement signals
- Transitions: learning dynamics (from Kalman model)
- Rewards: multi-objective (coverage, engagement, not-overwhelming)

The belief state (distribution over hidden states) is maintained by
the Kalman filter from learner_model.state_space.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ads_agent.learner_model.state_space import LearnerState


class ActionType(Enum):
    """Types of actions the agent can take."""
    RECOMMEND_COURSE = "recommend_course"
    GIVE_ASSESSMENT = "give_assessment"
    SUGGEST_REST = "suggest_rest"
    REVIEW_MATERIAL = "review_material"


@dataclass(frozen=True)
class Action:
    """An action in the POMDP."""

    action_type: ActionType
    target_id: str = ""        # course_id or assessment_id
    competency_idx: int = -1   # which competency this targets
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class POMDPSpec:
    """POMDP specification for curriculum recommendation.

    Continuous state space (handled via belief state from Kalman filter).
    Discrete action space (enumerated course recommendations + assessments).
    """

    n_competencies: int
    actions: List[Action]
    gamma: float = 0.95          # discount factor
    planning_horizon: int = 10   # steps to plan ahead

    @property
    def n_actions(self) -> int:
        return len(self.actions)


@dataclass
class BeliefState:
    """Belief over hidden learner state.

    In our case, the belief IS the Kalman filter posterior:
    a Gaussian N(mean, covariance). This is the sufficient statistic
    for decision-making under partial observability.
    """

    mean: np.ndarray          # (K,) posterior mean
    covariance: np.ndarray    # (K, K) posterior covariance
    timestep: int = 0

    @property
    def entropy(self) -> float:
        """Differential entropy of the Gaussian belief.

        Higher entropy = more uncertainty = more need for exploration.
        """
        K = len(self.mean)
        sign, logdet = np.linalg.slogdet(self.covariance)
        if sign <= 0:
            return float("inf")
        return 0.5 * (K * np.log(2 * np.pi * np.e) + logdet)

    @classmethod
    def from_learner_state(cls, state: LearnerState) -> "BeliefState":
        return cls(
            mean=state.mean.copy(),
            covariance=state.covariance.copy(),
            timestep=state.timestep,
        )


# ============================================================================
# Reward functions
# ============================================================================

@dataclass
class RewardWeights:
    """Multi-objective reward weights (user-configurable Pareto preferences)."""

    learning_progress: float = 0.4    # how much competency grew
    engagement: float = 0.2           # motivation - fatigue
    coverage: float = 0.2             # breadth across competencies
    information_gain: float = 0.2     # reduction in belief entropy

    def to_array(self) -> np.ndarray:
        return np.array([
            self.learning_progress,
            self.engagement,
            self.coverage,
            self.information_gain,
        ])


def compute_reward(
    belief_before: BeliefState,
    belief_after: BeliefState,
    action: Action,
    weights: RewardWeights,
) -> Tuple[float, Dict[str, float]]:
    """Compute multi-objective reward for a transition.

    Returns scalar reward and component breakdown.
    """
    K = len(belief_before.mean)
    n_comp = K - 2  # exclude motivation, fatigue

    # 1. Learning progress: increase in competency means
    progress = float(np.mean(belief_after.mean[:n_comp] - belief_before.mean[:n_comp]))
    progress = max(progress, 0.0)  # only count positive progress

    # 2. Engagement: motivation - fatigue (from belief mean)
    motivation_after = belief_after.mean[n_comp] if K > n_comp else 0.5
    fatigue_after = belief_after.mean[n_comp + 1] if K > n_comp + 1 else 0.0
    engagement = float(motivation_after - 0.5 * fatigue_after)

    # 3. Coverage: inverse of competency variance (more uniform = better)
    comp_means = belief_after.mean[:n_comp]
    coverage = 1.0 - float(np.std(comp_means)) if n_comp > 1 else 0.5

    # 4. Information gain: entropy reduction
    info_gain = max(belief_before.entropy - belief_after.entropy, 0.0)
    info_gain = min(info_gain, 5.0)  # cap to avoid domination
    info_gain /= 5.0  # normalize to [0, 1]

    components = {
        "learning_progress": progress,
        "engagement": engagement,
        "coverage": coverage,
        "information_gain": info_gain,
    }

    w = weights.to_array()
    vals = np.array([progress, engagement, coverage, info_gain])
    scalar_reward = float(w @ vals)

    return scalar_reward, components


# ============================================================================
# Simple heuristic policies (baselines)
# ============================================================================

def random_policy(
    belief: BeliefState,
    actions: List[Action],
    rng: np.random.RandomState,
) -> Action:
    """Uniformly random action selection (baseline)."""
    return actions[rng.randint(len(actions))]


def greedy_policy(
    belief: BeliefState,
    actions: List[Action],
) -> Action:
    """Greedy: recommend course targeting the weakest competency."""
    n_comp = len(belief.mean) - 2
    weakest = int(np.argmin(belief.mean[:n_comp]))

    # Find action targeting weakest competency
    for a in actions:
        if a.competency_idx == weakest and a.action_type == ActionType.RECOMMEND_COURSE:
            return a

    # Fallback: first course recommendation
    for a in actions:
        if a.action_type == ActionType.RECOMMEND_COURSE:
            return a
    return actions[0]


def exploration_policy(
    belief: BeliefState,
    actions: List[Action],
) -> Action:
    """Exploration: choose action targeting most uncertain competency."""
    uncertainties = np.diag(belief.covariance)
    n_comp = len(belief.mean) - 2
    most_uncertain = int(np.argmax(uncertainties[:n_comp]))

    # Prefer assessment for uncertain competencies
    for a in actions:
        if a.competency_idx == most_uncertain and a.action_type == ActionType.GIVE_ASSESSMENT:
            return a

    # Fall back to course for that competency
    for a in actions:
        if a.competency_idx == most_uncertain and a.action_type == ActionType.RECOMMEND_COURSE:
            return a
    return actions[0]


def fatigue_aware_policy(
    belief: BeliefState,
    actions: List[Action],
    fatigue_threshold: float = 0.6,
) -> Action:
    """Fatigue-aware: rest when fatigued, otherwise exploit."""
    fatigue = belief.mean[-1] if len(belief.mean) > 2 else 0.0

    if fatigue > fatigue_threshold:
        for a in actions:
            if a.action_type == ActionType.SUGGEST_REST:
                return a

    return greedy_policy(belief, actions)
