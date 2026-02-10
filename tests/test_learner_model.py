"""Tests for learner model: Kalman filter, simulator, evaluation."""
import numpy as np
import pytest

from ads_agent.learner_model.state_space import (
    LearnerState,
    KalmanSpec,
    KalmanFilter,
    build_competency_model,
)
from ads_agent.learner_model.simulator import (
    SimulatedTrace,
    simulate_learner,
    simulate_cohort,
    evaluate_filter,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_spec():
    """2-competency model (4 states: comp0, comp1, motivation, fatigue)."""
    return build_competency_model(n_competencies=2)


@pytest.fixture
def kf(simple_spec):
    return KalmanFilter(simple_spec)


# ============================================================================
# LearnerState
# ============================================================================

class TestLearnerState:
    def test_dim(self):
        state = LearnerState(mean=np.zeros(5), covariance=np.eye(5))
        assert state.dim == 5

    def test_uncertainty(self):
        cov = np.diag([1, 4, 9])
        state = LearnerState(mean=np.zeros(3), covariance=cov)
        np.testing.assert_allclose(state.uncertainty, [1, 2, 3])

    def test_confidence(self):
        # Low uncertainty -> high confidence
        cov = np.diag([0.01, 0.01])
        state = LearnerState(mean=np.array([0.5, 0.5]), covariance=cov)
        assert state.confidence(0) > 0.9

        # High uncertainty -> low confidence
        cov2 = np.diag([100, 100])
        state2 = LearnerState(mean=np.array([0.5, 0.5]), covariance=cov2)
        assert state2.confidence(0) < 0.1

    def test_to_dict(self):
        state = LearnerState(
            mean=np.array([0.5, 0.7]),
            covariance=np.eye(2) * 0.1,
            timestep=3,
            state_names=["skill_a", "skill_b"],
        )
        d = state.to_dict()
        assert d["timestep"] == 3
        assert "skill_a" in d["states"]
        assert "mean" in d["states"]["skill_a"]


# ============================================================================
# KalmanSpec
# ============================================================================

class TestKalmanSpec:
    def test_build_competency_model_dims(self, simple_spec):
        assert simple_spec.state_dim == 4  # 2 comp + motivation + fatigue
        assert simple_spec.obs_dim == 3    # 2 comp obs + engagement proxy
        assert simple_spec.action_dim == 3  # 2 comp actions + rest

    def test_state_names(self, simple_spec):
        assert len(simple_spec.state_names) == 4
        assert "motivation" in simple_spec.state_names
        assert "fatigue" in simple_spec.state_names

    def test_transition_matrix_shape(self, simple_spec):
        assert simple_spec.A.shape == (4, 4)
        assert simple_spec.B.shape == (4, 3)
        assert simple_spec.C.shape == (3, 4)
        assert simple_spec.Q.shape == (4, 4)
        assert simple_spec.R.shape == (3, 3)

    def test_competency_decay(self, simple_spec):
        # Diagonal of A for competencies should be < 1 (forgetting)
        for i in range(2):
            assert simple_spec.A[i, i] < 1.0

    def test_motivation_mean_reversion(self, simple_spec):
        assert simple_spec.A[2, 2] == 0.9  # persistence < 1

    def test_fatigue_recovery(self, simple_spec):
        assert simple_spec.A[3, 3] == 0.8  # recovery < 1


# ============================================================================
# KalmanFilter
# ============================================================================

class TestKalmanFilter:
    def test_initialize_default(self, kf):
        state = kf.initialize()
        assert state.dim == 4
        np.testing.assert_array_equal(state.mean, np.zeros(4))
        assert state.timestep == 0

    def test_initialize_custom(self, kf):
        prior = np.array([0.5, 0.5, 0.5, 0.0])
        state = kf.initialize(prior_mean=prior)
        np.testing.assert_array_equal(state.mean, prior)

    def test_predict_increments_timestep(self, kf):
        state = kf.initialize()
        predicted = kf.predict(state)
        assert predicted.timestep == 1

    def test_predict_covariance_formula(self, kf, simple_spec):
        """Predict covariance follows A P A' + Q."""
        state = kf.initialize()
        predicted = kf.predict(state)
        A, Q = simple_spec.A, simple_spec.Q
        expected_cov = A @ state.covariance @ A.T + Q
        np.testing.assert_allclose(predicted.covariance, expected_cov, atol=1e-10)

    def test_update_decreases_uncertainty(self, kf, simple_spec):
        state = kf.initialize()
        predicted = kf.predict(state)
        obs = np.array([0.3, 0.4, 0.5])
        updated = kf.update(predicted, obs)
        # Observation reduces uncertainty
        assert np.trace(updated.covariance) < np.trace(predicted.covariance)

    def test_step_combines_predict_update(self, kf):
        state = kf.initialize()
        obs = np.array([0.3, 0.4, 0.5])
        action = np.array([1.0, 0.0, 0.0])

        stepped = kf.step(state, obs, action)
        manually = kf.update(kf.predict(state, action), obs)

        np.testing.assert_allclose(stepped.mean, manually.mean, atol=1e-10)
        np.testing.assert_allclose(stepped.covariance, manually.covariance, atol=1e-10)

    def test_action_effect(self, kf, simple_spec):
        state = kf.initialize()
        # Action on comp_0 should increase predicted comp_0
        action_comp0 = np.array([1.0, 0.0, 0.0])
        pred_with_action = kf.predict(state, action_comp0)
        pred_no_action = kf.predict(state)
        # B[0, 0] = learning_rate > 0
        assert pred_with_action.mean[0] > pred_no_action.mean[0]

    def test_log_likelihood_finite(self, kf):
        state = kf.initialize()
        obs = np.array([0.3, 0.4, 0.5])
        ll = kf.log_likelihood(state, obs)
        assert np.isfinite(ll)

    def test_convergence_over_time(self, kf, simple_spec):
        """Filter should converge when fed consistent observations."""
        state = kf.initialize()
        true_val = np.array([0.6, 0.4, 0.6, 0.1])
        rng = np.random.RandomState(42)

        for t in range(50):
            obs = simple_spec.C @ true_val + rng.normal(0, 0.1, simple_spec.obs_dim)
            state = kf.step(state, obs)

        # After 50 observations, estimate should be close to true value
        assert np.abs(state.mean[0] - true_val[0]) < 0.3
        assert np.abs(state.mean[1] - true_val[1]) < 0.3


# ============================================================================
# RTS Smoother
# ============================================================================

class TestSmoother:
    def test_smooth_reduces_error(self, kf, simple_spec):
        """Smoother should be at least as good as filter."""
        rng = np.random.RandomState(42)
        true_val = np.array([0.5, 0.5, 0.6, 0.1])

        filtered = []
        state = kf.initialize()
        for t in range(20):
            obs = simple_spec.C @ true_val + rng.normal(0, 0.1, simple_spec.obs_dim)
            state = kf.step(state, obs)
            filtered.append(state)

        smoothed = kf.smooth(filtered)
        assert len(smoothed) == len(filtered)

        # Smoother uncertainty should be <= filter uncertainty (on average)
        filter_traces = [np.trace(s.covariance) for s in filtered]
        smooth_traces = [np.trace(s.covariance) for s in smoothed]
        assert np.mean(smooth_traces) <= np.mean(filter_traces) + 1e-10

    def test_smooth_single_state(self, kf):
        state = kf.initialize()
        smoothed = kf.smooth([state])
        assert len(smoothed) == 1


# ============================================================================
# Simulator
# ============================================================================

class TestSimulator:
    def test_simulate_learner_shape(self, simple_spec):
        trace = simulate_learner(simple_spec, n_steps=30)
        assert trace.true_states.shape == (30, 4)
        assert trace.observations.shape == (30, 3)
        assert trace.actions.shape == (30, 3)
        assert trace.timesteps == 30

    def test_simulate_policies(self, simple_spec):
        for policy in ["random", "sequential", "focused", "optimal_explore"]:
            trace = simulate_learner(simple_spec, n_steps=20, action_policy=policy)
            assert trace.true_states.shape[0] == 20
            # Actions should be one-hot (one action per step)
            for t in range(20):
                assert np.sum(trace.actions[t]) == pytest.approx(1.0)

    def test_simulate_deterministic_seed(self, simple_spec):
        t1 = simulate_learner(simple_spec, n_steps=10, seed=99)
        t2 = simulate_learner(simple_spec, n_steps=10, seed=99)
        np.testing.assert_array_equal(t1.true_states, t2.true_states)

    def test_states_bounded(self, simple_spec):
        trace = simulate_learner(simple_spec, n_steps=100)
        # Competencies, motivation, fatigue should be in [0, 1]
        assert np.all(trace.true_states >= -0.01)
        assert np.all(trace.true_states <= 1.01)

    def test_to_dict(self, simple_spec):
        trace = simulate_learner(simple_spec, n_steps=10, learner_id="test_001")
        d = trace.to_dict()
        assert d["learner_id"] == "test_001"
        assert d["timesteps"] == 10

    def test_simulate_cohort_size(self, simple_spec):
        cohort = simulate_cohort(simple_spec, n_learners=8, n_steps=10)
        assert len(cohort) == 8
        for trace in cohort:
            assert trace.timesteps == 10


# ============================================================================
# Filter Evaluation
# ============================================================================

class TestEvaluateFilter:
    def test_perfect_estimator(self):
        """If estimated = true, errors should be near-zero."""
        true_states = np.random.RandomState(42).rand(10, 3)
        estimated = [true_states[t] for t in range(10)]
        covs = [np.eye(3) * 0.01 for _ in range(10)]

        result = evaluate_filter(true_states, estimated, covs)
        assert result["overall_rmse"] < 1e-10
        assert result["mean_coverage"] == 1.0  # all within 2-sigma

    def test_poor_estimator(self):
        """Large constant error should yield bad RMSE."""
        true_states = np.zeros((10, 3))
        estimated = [np.ones(3) * 5.0 for _ in range(10)]
        covs = [np.eye(3) * 0.01 for _ in range(10)]

        result = evaluate_filter(true_states, estimated, covs)
        assert result["overall_rmse"] > 4.0
        assert result["mean_coverage"] == 0.0  # nothing within 2*0.1

    def test_nees_calibration(self, kf, simple_spec):
        """NEES should be ~K for well-calibrated filter."""
        trace = simulate_learner(simple_spec, n_steps=50, seed=42)

        # Run filter on simulated data
        state = kf.initialize()
        estimated_states = []
        estimated_covs = []
        for t in range(50):
            state = kf.step(state, trace.observations[t], trace.actions[t])
            estimated_states.append(state.mean.copy())
            estimated_covs.append(state.covariance.copy())

        result = evaluate_filter(trace.true_states, estimated_states, estimated_covs)
        # NEES should be roughly state_dim (4)
        assert result["mean_nees"] < 20  # generous bound
        assert result["calibrated"] or result["mean_nees"] < 10
