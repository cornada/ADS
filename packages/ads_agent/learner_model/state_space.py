"""State-space model for latent learner competency tracking.

Models learner competency as a hidden state evolving over time,
observed only through noisy signals (test scores, task completion,
time-on-task). Uses Kalman filtering for Gaussian dynamics and
particle filtering for nonlinear/non-Gaussian cases.

State vector (per learner):
  x_t = [competency_1, ..., competency_K, motivation, fatigue]

Dynamics:
  x_{t+1} = A * x_t + B * action_t + process_noise

Observation:
  y_t = C * x_t + observation_noise

Educational interpretation:
- Competency dimensions correspond to FGOS competency groups (УК/ОПК/ПК)
- Motivation is a latent drive that modulates learning rate
- Fatigue accumulates and recovers (mean-reverting)
- Actions = course/task assignments from the agent
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class LearnerState:
    """Current belief about a learner's latent state."""

    mean: np.ndarray          # (K,) posterior mean
    covariance: np.ndarray    # (K, K) posterior covariance
    timestep: int = 0
    state_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.mean)

    @property
    def uncertainty(self) -> np.ndarray:
        """Per-dimension uncertainty (std dev)."""
        return np.sqrt(np.diag(self.covariance))

    def confidence(self, dim: int = 0) -> float:
        """Confidence in a specific dimension (1 - normalized uncertainty)."""
        std = self.uncertainty[dim]
        return float(1.0 / (1.0 + std))

    def to_dict(self) -> Dict[str, Any]:
        names = self.state_names or [f"dim_{i}" for i in range(self.dim)]
        return {
            "timestep": self.timestep,
            "states": {
                name: {
                    "mean": round(float(self.mean[i]), 4),
                    "std": round(float(self.uncertainty[i]), 4),
                    "confidence": round(self.confidence(i), 4),
                }
                for i, name in enumerate(names)
            },
            "metadata": self.metadata,
        }


@dataclass
class KalmanSpec:
    """Specification of a linear-Gaussian state-space model."""

    A: np.ndarray       # (K, K) state transition
    B: np.ndarray       # (K, M) action effect matrix
    C: np.ndarray       # (D, K) observation matrix
    Q: np.ndarray       # (K, K) process noise covariance
    R: np.ndarray       # (D, D) observation noise covariance
    state_names: List[str] = field(default_factory=list)

    @property
    def state_dim(self) -> int:
        return self.A.shape[0]

    @property
    def obs_dim(self) -> int:
        return self.C.shape[0]

    @property
    def action_dim(self) -> int:
        return self.B.shape[1]


class KalmanFilter:
    """Kalman filter for linear-Gaussian learner state tracking.

    Standard predict-update cycle:
    - predict: propagate state forward using dynamics model
    - update: incorporate new observation to refine estimate
    """

    def __init__(self, spec: KalmanSpec) -> None:
        self.spec = spec

    def initialize(
        self,
        prior_mean: Optional[np.ndarray] = None,
        prior_cov: Optional[np.ndarray] = None,
    ) -> LearnerState:
        """Create initial learner state."""
        K = self.spec.state_dim
        mean = prior_mean if prior_mean is not None else np.zeros(K)
        cov = prior_cov if prior_cov is not None else np.eye(K) * 1.0
        return LearnerState(
            mean=mean.copy(),
            covariance=cov.copy(),
            timestep=0,
            state_names=list(self.spec.state_names),
        )

    def predict(
        self,
        state: LearnerState,
        action: Optional[np.ndarray] = None,
    ) -> LearnerState:
        """Predict next state (time update)."""
        A, B, Q = self.spec.A, self.spec.B, self.spec.Q

        # State prediction
        mean_pred = A @ state.mean
        if action is not None:
            mean_pred += B @ action

        # Covariance prediction
        cov_pred = A @ state.covariance @ A.T + Q

        return LearnerState(
            mean=mean_pred,
            covariance=cov_pred,
            timestep=state.timestep + 1,
            state_names=state.state_names,
        )

    def update(
        self,
        state: LearnerState,
        observation: np.ndarray,
    ) -> LearnerState:
        """Update state with observation (measurement update)."""
        C, R = self.spec.C, self.spec.R

        # Innovation
        y_pred = C @ state.mean
        innovation = observation - y_pred

        # Innovation covariance
        S = C @ state.covariance @ C.T + R

        # Kalman gain
        K = state.covariance @ C.T @ np.linalg.inv(S)

        # Posterior
        mean_post = state.mean + K @ innovation
        cov_post = (np.eye(self.spec.state_dim) - K @ C) @ state.covariance

        return LearnerState(
            mean=mean_post,
            covariance=cov_post,
            timestep=state.timestep,
            state_names=state.state_names,
            metadata={"innovation": float(np.linalg.norm(innovation))},
        )

    def step(
        self,
        state: LearnerState,
        observation: np.ndarray,
        action: Optional[np.ndarray] = None,
    ) -> LearnerState:
        """Full predict + update cycle."""
        predicted = self.predict(state, action)
        return self.update(predicted, observation)

    def log_likelihood(
        self,
        state: LearnerState,
        observation: np.ndarray,
    ) -> float:
        """Log-likelihood of observation given current state."""
        C, R = self.spec.C, self.spec.R
        y_pred = C @ state.mean
        S = C @ state.covariance @ C.T + R
        diff = observation - y_pred
        d = len(observation)
        sign, logdet = np.linalg.slogdet(S)
        ll = -0.5 * (d * np.log(2 * np.pi) + logdet + diff @ np.linalg.inv(S) @ diff)
        return float(ll)

    def smooth(
        self,
        states: List[LearnerState],
    ) -> List[LearnerState]:
        """Rauch-Tung-Striebel smoother (backward pass).

        Given a sequence of filtered states, compute smoothed estimates
        that use all observations (past AND future).
        """
        A, Q = self.spec.A, self.spec.Q
        n = len(states)
        if n <= 1:
            return states

        smoothed = [None] * n
        smoothed[-1] = states[-1]

        for t in range(n - 2, -1, -1):
            # Predicted state at t+1 from filtered state at t
            mean_pred = A @ states[t].mean
            cov_pred = A @ states[t].covariance @ A.T + Q

            # Smoother gain
            G = states[t].covariance @ A.T @ np.linalg.inv(cov_pred)

            # Smoothed estimates
            mean_smooth = states[t].mean + G @ (smoothed[t + 1].mean - mean_pred)
            cov_smooth = states[t].covariance + G @ (
                smoothed[t + 1].covariance - cov_pred
            ) @ G.T

            smoothed[t] = LearnerState(
                mean=mean_smooth,
                covariance=cov_smooth,
                timestep=states[t].timestep,
                state_names=states[t].state_names,
            )

        return smoothed


def build_competency_model(
    n_competencies: int = 3,
    learning_rate: float = 0.1,
    decay_rate: float = 0.02,
    process_noise: float = 0.05,
    obs_noise: float = 0.3,
    competency_names: Optional[List[str]] = None,
) -> KalmanSpec:
    """Build a standard competency tracking model.

    State = [comp_1, ..., comp_K, motivation, fatigue]
    - Competencies grow via learning (action-dependent)
    - Motivation is mean-reverting around 0.5
    - Fatigue accumulates during tasks, recovers between sessions

    Args:
        n_competencies: Number of competency dimensions
        learning_rate: How fast competencies grow per unit action
        decay_rate: Forgetting/fatigue accumulation rate
        process_noise: State evolution noise
        obs_noise: Observation noise
        competency_names: Names for competency dimensions

    Returns:
        KalmanSpec for the model
    """
    K = n_competencies + 2  # competencies + motivation + fatigue
    M = n_competencies + 1  # one action per competency + rest action

    # State transition: A
    A = np.eye(K)
    # Competencies: slight decay (forgetting)
    for i in range(n_competencies):
        A[i, i] = 1.0 - decay_rate
    # Motivation: mean-reverting to 0.5
    A[n_competencies, n_competencies] = 0.9  # persistence
    # Fatigue: mean-reverting to 0 (recovery)
    A[n_competencies + 1, n_competencies + 1] = 0.8

    # Action effect: B
    B = np.zeros((K, M))
    # Each action increases corresponding competency
    for i in range(n_competencies):
        B[i, i] = learning_rate
    # Motivation effect from engagement
    B[n_competencies, :n_competencies] = 0.02  # any action slightly boosts motivation
    # Fatigue from any task
    B[n_competencies + 1, :n_competencies] = 0.05
    # Rest action reduces fatigue
    B[n_competencies + 1, -1] = -0.15

    # Observation: C (observe competencies noisily, motivation indirectly)
    D = n_competencies + 1  # observe each competency + one engagement proxy
    C = np.zeros((D, K))
    # Direct competency observations (test scores)
    for i in range(n_competencies):
        C[i, i] = 1.0
    # Engagement proxy = f(motivation - fatigue)
    C[n_competencies, n_competencies] = 1.0      # motivation
    C[n_competencies, n_competencies + 1] = -0.5  # fatigue dampens

    # Noise
    Q = np.eye(K) * process_noise ** 2
    R = np.eye(D) * obs_noise ** 2

    names = competency_names or [f"comp_{i}" for i in range(n_competencies)]
    names += ["motivation", "fatigue"]

    return KalmanSpec(A=A, B=B, C=C, Q=Q, R=R, state_names=names)
