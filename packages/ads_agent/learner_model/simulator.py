"""Synthetic learner trace simulator.

Generates realistic learner interaction traces with known ground truth,
enabling validation of the state-space model. Each trace represents
a learner progressing through a curriculum with observable signals.

Simulation parameters control:
- Learning speed (individual differences)
- Motivation dynamics (engagement/burnout cycles)
- Observation noise (assessment reliability)
- Action sequences (curriculum choices)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .state_space import KalmanSpec


@dataclass
class SimulatedTrace:
    """A complete simulated learner trajectory."""

    true_states: np.ndarray     # (T, K) ground truth latent states
    observations: np.ndarray    # (T, D) noisy observations
    actions: np.ndarray         # (T, M) actions taken
    timesteps: int
    state_dim: int
    obs_dim: int
    learner_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "timesteps": self.timesteps,
            "state_dim": self.state_dim,
            "obs_dim": self.obs_dim,
            "final_state": self.true_states[-1].tolist(),
            "metadata": self.metadata,
        }


def simulate_learner(
    spec: KalmanSpec,
    n_steps: int = 50,
    initial_state: Optional[np.ndarray] = None,
    action_policy: str = "random",
    seed: int = 42,
    learner_id: str = "",
) -> SimulatedTrace:
    """Simulate a single learner trajectory.

    Args:
        spec: State-space model specification
        n_steps: Number of time steps
        initial_state: Starting state (default: zeros)
        action_policy: "random", "sequential", "focused"
        seed: Random seed
        learner_id: Identifier for this learner

    Returns:
        SimulatedTrace with ground truth and observations
    """
    rng = np.random.RandomState(seed)
    K = spec.state_dim
    D = spec.obs_dim
    M = spec.action_dim

    # Initial state
    x = initial_state.copy() if initial_state is not None else np.zeros(K)
    # Start with some baseline competency
    x[:K - 2] = rng.uniform(0.1, 0.3, size=K - 2)
    x[K - 2] = 0.5  # motivation
    x[K - 1] = 0.0  # fatigue

    true_states = np.zeros((n_steps, K))
    observations = np.zeros((n_steps, D))
    actions = np.zeros((n_steps, M))

    for t in range(n_steps):
        # Choose action based on policy
        action = _choose_action(spec, x, t, action_policy, rng)
        actions[t] = action

        # State evolution
        process_noise = rng.multivariate_normal(np.zeros(K), spec.Q)
        x = spec.A @ x + spec.B @ action + process_noise

        # Clamp states to reasonable ranges
        x[:K - 2] = np.clip(x[:K - 2], 0, 1)  # competencies [0, 1]
        x[K - 2] = np.clip(x[K - 2], 0, 1)     # motivation [0, 1]
        x[K - 1] = np.clip(x[K - 1], 0, 1)     # fatigue [0, 1]

        true_states[t] = x

        # Observation
        obs_noise = rng.multivariate_normal(np.zeros(D), spec.R)
        observations[t] = spec.C @ x + obs_noise

    return SimulatedTrace(
        true_states=true_states,
        observations=observations,
        actions=actions,
        timesteps=n_steps,
        state_dim=K,
        obs_dim=D,
        learner_id=learner_id,
        metadata={"action_policy": action_policy, "seed": seed},
    )


def _choose_action(
    spec: KalmanSpec,
    state: np.ndarray,
    timestep: int,
    policy: str,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Choose action based on policy."""
    M = spec.action_dim
    action = np.zeros(M)
    n_comp = M - 1  # last action = rest

    if policy == "random":
        # Random task or rest
        idx = rng.randint(0, M)
        action[idx] = 1.0

    elif policy == "sequential":
        # Cycle through competencies, rest every 5th step
        if timestep % 5 == 4:
            action[-1] = 1.0  # rest
        else:
            idx = timestep % n_comp
            action[idx] = 1.0

    elif policy == "focused":
        # Focus on weakest competency, rest when fatigued
        fatigue = state[-1] if len(state) > 1 else 0
        if fatigue > 0.6:
            action[-1] = 1.0  # rest
        else:
            comps = state[:n_comp]
            weakest = int(np.argmin(comps))
            action[weakest] = 1.0

    elif policy == "optimal_explore":
        # Explore-exploit: random early, focused late
        if timestep < 10:
            idx = rng.randint(0, M)
            action[idx] = 1.0
        else:
            fatigue = state[-1]
            if fatigue > 0.5:
                action[-1] = 1.0
            else:
                comps = state[:n_comp]
                weakest = int(np.argmin(comps))
                action[weakest] = 1.0

    return action


def simulate_cohort(
    spec: KalmanSpec,
    n_learners: int = 20,
    n_steps: int = 50,
    policies: Optional[List[str]] = None,
    base_seed: int = 42,
) -> List[SimulatedTrace]:
    """Simulate a cohort of learners with different characteristics.

    Creates learners with varying initial states and learning speeds,
    simulating individual differences in a classroom.
    """
    if policies is None:
        policies = ["random", "sequential", "focused", "optimal_explore"]

    traces = []
    for i in range(n_learners):
        policy = policies[i % len(policies)]
        seed = base_seed + i

        trace = simulate_learner(
            spec,
            n_steps=n_steps,
            action_policy=policy,
            seed=seed,
            learner_id=f"learner_{i:03d}",
        )
        traces.append(trace)

    return traces


def evaluate_filter(
    true_states: np.ndarray,
    estimated_states: List[np.ndarray],
    estimated_covs: List[np.ndarray],
) -> Dict[str, Any]:
    """Evaluate filter performance against ground truth.

    Args:
        true_states: (T, K) ground truth
        estimated_states: List of (K,) posterior means
        estimated_covs: List of (K, K) posterior covariances

    Returns:
        Performance metrics
    """
    T = true_states.shape[0]
    K = true_states.shape[1]

    est_arr = np.array(estimated_states)
    errors = true_states - est_arr

    # RMSE per dimension
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))

    # Mean absolute error
    mae = np.mean(np.abs(errors), axis=0)

    # Coverage: fraction of time true state is within 2-sigma of estimate
    coverage = np.zeros(K)
    for t in range(T):
        stds = np.sqrt(np.diag(estimated_covs[t]))
        within = np.abs(errors[t]) < 2 * stds
        coverage += within
    coverage /= T

    # Normalized estimation error squared (NEES) — calibration check
    # Should be ~K if filter is well-calibrated
    nees_values = []
    for t in range(T):
        err = errors[t]
        P = estimated_covs[t]
        try:
            nees = float(err @ np.linalg.inv(P) @ err)
            nees_values.append(nees)
        except np.linalg.LinAlgError:
            pass

    mean_nees = float(np.mean(nees_values)) if nees_values else float("nan")

    return {
        "rmse_per_dim": rmse.tolist(),
        "mae_per_dim": mae.tolist(),
        "overall_rmse": float(np.mean(rmse)),
        "coverage_2sigma": coverage.tolist(),
        "mean_coverage": float(np.mean(coverage)),
        "mean_nees": mean_nees,
        "expected_nees": K,
        "calibrated": abs(mean_nees - K) < K * 0.5,  # within 50% of expected
    }
