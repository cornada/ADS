"""Learner API endpoints.

Manages learner profiles, competency tracking via Kalman filter,
and course recommendations from the POMDP controller.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import numpy as np
import threading

from ads_agent.learner_model.state_space import (
    KalmanFilter, KalmanSpec, LearnerState, build_competency_model,
)
from ads_agent.controller.pomdp_spec import (
    Action, ActionType, BeliefState, RewardWeights,
    compute_reward, greedy_policy, exploration_policy,
    fatigue_aware_policy,
)

router = APIRouter()


# ============================================================================
# Schemas
# ============================================================================

class LearnerProfile(BaseModel):
    """Learner profile with current competency estimate."""
    learner_id: str
    timestep: int
    competencies: Dict[str, float]
    uncertainties: Dict[str, float]
    motivation: float
    fatigue: float
    confidence: float


class ObservationInput(BaseModel):
    """New observation (assessment result or engagement signal)."""
    learner_id: str
    scores: List[float] = Field(..., description="Observed signals (test scores, engagement)")
    action_taken: Optional[List[float]] = Field(None, description="Action vector (one-hot)")


class RecommendationRequest(BaseModel):
    """Request for course recommendations."""
    learner_id: str
    policy: str = Field(default="fatigue_aware", description="greedy | exploration | fatigue_aware")
    n_recommendations: int = Field(default=3, ge=1, le=10)


class Recommendation(BaseModel):
    """A course recommendation."""
    action_type: str
    target_id: str
    competency_focus: int
    reason: str
    expected_reward: float


class RecommendationResponse(BaseModel):
    """Response with recommendations and current state."""
    learner_id: str
    profile: LearnerProfile
    recommendations: List[Recommendation]


# ============================================================================
# Learner session store
# ============================================================================

_lock = threading.Lock()
_sessions: Dict[str, LearnerState] = {}
_kf: Optional[KalmanFilter] = None
_spec: Optional[KalmanSpec] = None
_actions: List[Action] = []


def _ensure_model():
    """Lazy-initialize the competency model."""
    global _kf, _spec, _actions
    if _kf is not None:
        return

    _spec = build_competency_model(
        n_competencies=3,
        competency_names=["programming", "algorithms", "systems"],
    )
    _kf = KalmanFilter(_spec)
    _actions = [
        Action(ActionType.RECOMMEND_COURSE, "programming_course", competency_idx=0),
        Action(ActionType.RECOMMEND_COURSE, "algorithms_course", competency_idx=1),
        Action(ActionType.RECOMMEND_COURSE, "systems_course", competency_idx=2),
        Action(ActionType.GIVE_ASSESSMENT, "diagnostic_test_0", competency_idx=0),
        Action(ActionType.GIVE_ASSESSMENT, "diagnostic_test_1", competency_idx=1),
        Action(ActionType.GIVE_ASSESSMENT, "diagnostic_test_2", competency_idx=2),
        Action(ActionType.SUGGEST_REST, "rest"),
    ]


def _state_to_profile(learner_id: str, state: LearnerState) -> LearnerProfile:
    """Convert internal state to API profile."""
    names = state.state_names or [f"comp_{i}" for i in range(state.dim)]
    n_comp = state.dim - 2

    competencies = {names[i]: round(float(state.mean[i]), 4) for i in range(n_comp)}
    uncertainties = {names[i]: round(float(state.uncertainty[i]), 4) for i in range(n_comp)}

    return LearnerProfile(
        learner_id=learner_id,
        timestep=state.timestep,
        competencies=competencies,
        uncertainties=uncertainties,
        motivation=round(float(state.mean[n_comp]), 4),
        fatigue=round(float(state.mean[n_comp + 1]), 4),
        confidence=round(float(np.mean([state.confidence(i) for i in range(n_comp)])), 4),
    )


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/register", response_model=LearnerProfile)
def register_learner(learner_id: str):
    """Register a new learner and initialize their state."""
    _ensure_model()

    with _lock:
        if learner_id in _sessions:
            raise HTTPException(status_code=409, detail=f"Learner already registered: {learner_id}")
        state = _kf.initialize()
        _sessions[learner_id] = state

    return _state_to_profile(learner_id, state)


@router.get("/{learner_id}", response_model=LearnerProfile)
def get_learner(learner_id: str):
    """Get current learner profile."""
    _ensure_model()

    with _lock:
        state = _sessions.get(learner_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Learner not found: {learner_id}")

    return _state_to_profile(learner_id, state)


@router.post("/observe", response_model=LearnerProfile)
def record_observation(obs: ObservationInput):
    """Record a new observation and update learner state."""
    _ensure_model()

    with _lock:
        state = _sessions.get(obs.learner_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Learner not found: {obs.learner_id}")

    observation = np.array(obs.scores)
    if len(observation) != _spec.obs_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {_spec.obs_dim} observations, got {len(observation)}",
        )

    action = np.array(obs.action_taken) if obs.action_taken else None
    if action is not None and len(action) != _spec.action_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {_spec.action_dim} action dims, got {len(action)}",
        )

    new_state = _kf.step(state, observation, action)

    with _lock:
        _sessions[obs.learner_id] = new_state

    return _state_to_profile(obs.learner_id, new_state)


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest):
    """Get personalized recommendations for a learner."""
    _ensure_model()

    with _lock:
        state = _sessions.get(req.learner_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Learner not found: {req.learner_id}")

    belief = BeliefState.from_learner_state(state)
    weights = RewardWeights()

    # Select policy
    recommendations = []
    used_actions = set()

    for _ in range(req.n_recommendations):
        available = [a for a in _actions if a.target_id not in used_actions]
        if not available:
            break

        if req.policy == "greedy":
            action = greedy_policy(belief, available)
        elif req.policy == "exploration":
            action = exploration_policy(belief, available)
        else:
            action = fatigue_aware_policy(belief, available)

        used_actions.add(action.target_id)

        # Estimate reward (using current belief as approximation)
        reason = _explain_recommendation(belief, action)

        recommendations.append(Recommendation(
            action_type=action.action_type.value,
            target_id=action.target_id,
            competency_focus=action.competency_idx,
            reason=reason,
            expected_reward=0.0,  # would require forward simulation
        ))

    return RecommendationResponse(
        learner_id=req.learner_id,
        profile=_state_to_profile(req.learner_id, state),
        recommendations=recommendations,
    )


def _explain_recommendation(belief: BeliefState, action: Action) -> str:
    """Generate human-readable explanation for a recommendation."""
    n_comp = len(belief.mean) - 2
    fatigue = belief.mean[-1]
    uncertainties = np.diag(belief.covariance)

    if action.action_type == ActionType.SUGGEST_REST:
        return f"Fatigue level is {fatigue:.0%} — rest recommended to improve future learning"

    if action.action_type == ActionType.GIVE_ASSESSMENT:
        unc = uncertainties[action.competency_idx]
        return f"High uncertainty ({unc:.2f}) in competency {action.competency_idx} — diagnostic assessment recommended"

    # Course recommendation
    level = belief.mean[action.competency_idx]
    return f"Competency {action.competency_idx} at {level:.0%} — targeted practice recommended"


@router.get("/list/active")
def list_active_learners():
    """List all active learner sessions."""
    _ensure_model()

    with _lock:
        learner_ids = list(_sessions.keys())

    return {
        "active_learners": len(learner_ids),
        "learner_ids": learner_ids,
    }
