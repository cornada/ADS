"""Tests for POMDP controller module."""
import numpy as np
import pytest

from ads_agent.controller.pomdp_spec import (
    Action,
    ActionType,
    BeliefState,
    POMDPSpec,
    RewardWeights,
    compute_reward,
    random_policy,
    greedy_policy,
    exploration_policy,
    fatigue_aware_policy,
)
from ads_agent.learner_model.state_space import LearnerState


# ============================================================================
# BeliefState
# ============================================================================

class TestBeliefState:
    def test_entropy_positive(self):
        belief = BeliefState(mean=np.zeros(3), covariance=np.eye(3))
        assert belief.entropy > 0

    def test_entropy_increases_with_uncertainty(self):
        b_low = BeliefState(mean=np.zeros(3), covariance=np.eye(3) * 0.1)
        b_high = BeliefState(mean=np.zeros(3), covariance=np.eye(3) * 10.0)
        assert b_high.entropy > b_low.entropy

    def test_from_learner_state(self):
        ls = LearnerState(
            mean=np.array([0.5, 0.5, 0.6, 0.1]),
            covariance=np.eye(4) * 0.2,
            timestep=5,
        )
        belief = BeliefState.from_learner_state(ls)
        np.testing.assert_array_equal(belief.mean, ls.mean)
        assert belief.timestep == 5


# ============================================================================
# Reward
# ============================================================================

class TestReward:
    @pytest.fixture
    def beliefs(self):
        before = BeliefState(
            mean=np.array([0.3, 0.4, 0.5, 0.2]),  # comp0, comp1, motivation, fatigue
            covariance=np.eye(4) * 0.5,
        )
        after = BeliefState(
            mean=np.array([0.4, 0.5, 0.6, 0.3]),  # improvement
            covariance=np.eye(4) * 0.3,             # less uncertain
        )
        return before, after

    def test_reward_is_scalar(self, beliefs):
        before, after = beliefs
        action = Action(ActionType.RECOMMEND_COURSE, competency_idx=0)
        weights = RewardWeights()
        reward, components = compute_reward(before, after, action, weights)
        assert isinstance(reward, float)
        assert isinstance(components, dict)

    def test_reward_positive_for_progress(self, beliefs):
        before, after = beliefs
        action = Action(ActionType.RECOMMEND_COURSE, competency_idx=0)
        weights = RewardWeights(learning_progress=1.0, engagement=0.0, coverage=0.0, information_gain=0.0)
        reward, _ = compute_reward(before, after, action, weights)
        assert reward > 0  # competencies increased

    def test_information_gain_component(self, beliefs):
        before, after = beliefs
        action = Action(ActionType.GIVE_ASSESSMENT, competency_idx=0)
        weights = RewardWeights()
        _, components = compute_reward(before, after, action, weights)
        assert components["information_gain"] > 0  # entropy decreased

    def test_zero_weights(self, beliefs):
        before, after = beliefs
        action = Action(ActionType.RECOMMEND_COURSE)
        weights = RewardWeights(learning_progress=0.0, engagement=0.0, coverage=0.0, information_gain=0.0)
        reward, _ = compute_reward(before, after, action, weights)
        assert reward == 0.0


# ============================================================================
# Policies
# ============================================================================

class TestPolicies:
    @pytest.fixture
    def action_set(self):
        return [
            Action(ActionType.RECOMMEND_COURSE, target_id="c0", competency_idx=0),
            Action(ActionType.RECOMMEND_COURSE, target_id="c1", competency_idx=1),
            Action(ActionType.GIVE_ASSESSMENT, target_id="a0", competency_idx=0),
            Action(ActionType.GIVE_ASSESSMENT, target_id="a1", competency_idx=1),
            Action(ActionType.SUGGEST_REST, target_id="rest"),
        ]

    def test_random_policy(self, action_set):
        belief = BeliefState(mean=np.array([0.5, 0.5, 0.5, 0.1]), covariance=np.eye(4))
        rng = np.random.RandomState(42)
        action = random_policy(belief, action_set, rng)
        assert action in action_set

    def test_greedy_targets_weakest(self, action_set):
        # comp_0 = 0.2 (weak), comp_1 = 0.8 (strong)
        belief = BeliefState(mean=np.array([0.2, 0.8, 0.5, 0.1]), covariance=np.eye(4))
        action = greedy_policy(belief, action_set)
        assert action.competency_idx == 0

    def test_exploration_targets_uncertain(self, action_set):
        # High uncertainty on comp_1
        cov = np.diag([0.1, 5.0, 0.5, 0.5])
        belief = BeliefState(mean=np.array([0.5, 0.5, 0.5, 0.1]), covariance=cov)
        action = exploration_policy(belief, action_set)
        assert action.competency_idx == 1

    def test_fatigue_aware_rests_when_tired(self, action_set):
        # High fatigue
        belief = BeliefState(mean=np.array([0.5, 0.5, 0.5, 0.8]), covariance=np.eye(4))
        action = fatigue_aware_policy(belief, action_set, fatigue_threshold=0.6)
        assert action.action_type == ActionType.SUGGEST_REST

    def test_fatigue_aware_works_when_fresh(self, action_set):
        # Low fatigue → acts like greedy
        belief = BeliefState(mean=np.array([0.2, 0.8, 0.5, 0.1]), covariance=np.eye(4))
        action = fatigue_aware_policy(belief, action_set, fatigue_threshold=0.6)
        assert action.action_type == ActionType.RECOMMEND_COURSE
