"""Assessment API endpoints.

Provides adaptive testing (CAT) sessions via IRT,
ability estimation, and conformal prediction intervals.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import threading

from ads_agent.assessment.irt_model import (
    IRTModel, IRTItem, IRTResponse, AbilityEstimate, generate_item_bank,
)
from ads_agent.assessment.adaptive_selection import (
    AdaptiveSelector, CATSession,
)

router = APIRouter()


# ============================================================================
# Schemas
# ============================================================================

class ItemOut(BaseModel):
    """An assessment item to present to the learner."""
    item_id: str
    content_area: str
    difficulty_hint: str  # "easy", "medium", "hard" (not raw parameters)


class SessionInfo(BaseModel):
    """Current CAT session state."""
    session_id: str
    learner_id: str
    n_administered: int
    current_theta: float
    current_se: float
    confidence_interval: List[float]
    terminated: bool
    termination_reason: str


class ResponseInput(BaseModel):
    """Learner's response to an item."""
    session_id: str
    item_id: str
    correct: bool
    response_time: Optional[float] = None


class StartSessionRequest(BaseModel):
    """Request to start a new CAT session."""
    learner_id: str
    max_items: int = Field(default=25, ge=5, le=50)
    se_threshold: float = Field(default=0.3, ge=0.1, le=1.0)
    content_area: Optional[str] = None  # filter to specific competency


class NextItemResponse(BaseModel):
    """Response with the next item to administer."""
    item: Optional[ItemOut] = None
    session: SessionInfo
    message: str


# ============================================================================
# Session store
# ============================================================================

_lock = threading.Lock()
_cat_sessions: Dict[str, tuple[CATSession, AdaptiveSelector]] = {}
_model: Optional[IRTModel] = None
_items: List[IRTItem] = []


def _ensure_model():
    global _model, _items
    if _model is not None:
        return
    _items = generate_item_bank(n_items=50, n_areas=3, seed=42)
    _model = IRTModel(_items)


def _difficulty_hint(item: IRTItem) -> str:
    if item.difficulty < -0.5:
        return "easy"
    elif item.difficulty > 0.5:
        return "hard"
    return "medium"


def _session_info(session_id: str, session: CATSession) -> SessionInfo:
    est = session.current_estimate
    ci = est.confidence_interval if est else (0.0, 0.0)
    return SessionInfo(
        session_id=session_id,
        learner_id=session.learner_id,
        n_administered=session.n_administered,
        current_theta=est.theta if est else 0.0,
        current_se=est.se if est else 1.0,
        confidence_interval=list(ci),
        terminated=session.terminated,
        termination_reason=session.termination_reason,
    )


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/start", response_model=NextItemResponse)
def start_session(req: StartSessionRequest):
    """Start a new adaptive testing session."""
    _ensure_model()

    session_id = f"cat_{req.learner_id}_{len(_cat_sessions)}"

    pool = _items
    if req.content_area:
        pool = [i for i in _items if i.content_area == req.content_area]
        if len(pool) < 5:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough items for area '{req.content_area}' (need ≥5, have {len(pool)})",
            )

    selector = AdaptiveSelector(
        model=_model,
        item_pool=pool,
        criterion="mfi",
        max_items=req.max_items,
        se_threshold=req.se_threshold,
    )

    session = selector.start_session(req.learner_id)

    with _lock:
        _cat_sessions[session_id] = (session, selector)

    # Select first item
    result = selector.select_next(session)
    if result is None:
        return NextItemResponse(
            item=None,
            session=_session_info(session_id, session),
            message="No items available",
        )

    return NextItemResponse(
        item=ItemOut(
            item_id=result.selected_item.item_id,
            content_area=result.selected_item.content_area,
            difficulty_hint=_difficulty_hint(result.selected_item),
        ),
        session=_session_info(session_id, session),
        message=f"Session started. Administer item {result.selected_item.item_id}",
    )


@router.post("/respond", response_model=NextItemResponse)
def record_response(resp: ResponseInput):
    """Record a response and get the next item."""
    with _lock:
        entry = _cat_sessions.get(resp.session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {resp.session_id}")

    session, selector = entry

    if session.terminated:
        return NextItemResponse(
            item=None,
            session=_session_info(resp.session_id, session),
            message=f"Session already terminated: {session.termination_reason}",
        )

    # Record response
    selector.record_response(
        session, resp.item_id, resp.correct, resp.response_time,
    )

    # Check if terminated after recording
    if session.terminated:
        return NextItemResponse(
            item=None,
            session=_session_info(resp.session_id, session),
            message=f"Assessment complete ({session.termination_reason}). θ̂ = {session.current_estimate.theta:.2f}",
        )

    # Select next item
    result = selector.select_next(session)
    if result is None:
        return NextItemResponse(
            item=None,
            session=_session_info(resp.session_id, session),
            message="No more items available",
        )

    return NextItemResponse(
        item=ItemOut(
            item_id=result.selected_item.item_id,
            content_area=result.selected_item.content_area,
            difficulty_hint=_difficulty_hint(result.selected_item),
        ),
        session=_session_info(resp.session_id, session),
        message=f"Item {session.n_administered}/{selector.max_items}",
    )


@router.get("/{session_id}", response_model=SessionInfo)
def get_session(session_id: str):
    """Get current session state."""
    with _lock:
        entry = _cat_sessions.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    session, _ = entry
    return _session_info(session_id, session)


@router.get("/{session_id}/history")
def get_session_history(session_id: str):
    """Get full history of a CAT session."""
    with _lock:
        entry = _cat_sessions.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    session, _ = entry
    history = []
    for i, est in enumerate(session.ability_history):
        history.append({
            "step": i,
            "theta": est.theta,
            "se": est.se,
            "n_items": est.n_items,
        })

    return {
        "session_id": session_id,
        "learner_id": session.learner_id,
        "terminated": session.terminated,
        "termination_reason": session.termination_reason,
        "history": history,
        "responses": [
            {"item_id": r.item_id, "correct": r.correct}
            for r in session.administered
        ],
    }
