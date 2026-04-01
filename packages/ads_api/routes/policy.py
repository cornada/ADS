"""Policy API endpoints for C09 - Product-mode control loop.

Provides endpoints for:
- Recording selected Pareto options as policy decisions
- Retrieving policy history
- Exporting decisions for audit/reproducibility

A Policy represents a decision point: the user selects a specific Pareto-optimal
option given their current objective weights, constraints, and configuration.
This allows tracking of decisions over time for analysis and audit.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import hashlib
import threading

router = APIRouter()


# ============================================================================
# Policy Schema
# ============================================================================

class PolicyConfig(BaseModel):
    """Configuration snapshot at time of decision.

    Captures the full context needed to reproduce the decision:
    - Which dataset and embeddings were used
    - What objectives were enabled
    - Lens mode and constraint settings
    """
    dataset_id: str = Field(..., description="Dataset used for evaluation")
    embedding_kind: str = Field(default="stub", description="Embedding type")
    objectives: List[str] = Field(..., description="Enabled objective names")
    lens_mode: str = Field(default="identity", description="Lens mode: identity, diagonal, learned")
    autonomy_tau: float = Field(default=0.25, description="Autonomy constraint threshold")


class PolicyDecision(BaseModel):
    """A recorded policy decision.

    Represents a user selecting a specific option from the Pareto front.
    The decision captures the full context for audit and reproducibility.
    """
    policy_id: str = Field(..., description="Unique policy identifier")
    session_id: Optional[str] = Field(default=None, description="Session that made this decision")

    # Selected option details
    option_id: str = Field(..., description="Selected Pareto option ID")
    option_text: str = Field(default="", description="Text/description of selected option")

    # Objective scores at time of decision
    objective_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Objective scores for selected option"
    )

    # Was this option on the Pareto front?
    is_pareto: bool = Field(default=True, description="Whether option was Pareto-optimal")

    # Configuration snapshot
    config: PolicyConfig = Field(..., description="Configuration at time of decision")
    config_hash: str = Field(default="", description="Hash of config for quick comparison")

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional rationale from user
    rationale: Optional[str] = Field(default=None, description="User-provided decision rationale")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PolicySelectRequest(BaseModel):
    """Request to record a policy selection."""
    session_id: Optional[str] = None
    option_id: str = Field(..., description="ID of selected option")
    option_text: str = Field(default="", description="Text of selected option")
    objective_scores: Dict[str, float] = Field(default_factory=dict)
    is_pareto: bool = Field(default=True)
    config: PolicyConfig
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PolicySelectResponse(BaseModel):
    """Response after recording a policy selection."""
    ok: bool
    policy: PolicyDecision
    message: str


class PolicyHistoryResponse(BaseModel):
    """Response containing policy history."""
    policies: List[PolicyDecision]
    total_count: int
    filtered_count: int


class PolicyStatsResponse(BaseModel):
    """Statistics about policy decisions."""
    total_decisions: int
    unique_sessions: int
    decisions_by_dataset: Dict[str, int]
    decisions_by_option: Dict[str, int]
    most_selected_options: List[Dict[str, Any]]
    date_range: Optional[Dict[str, str]]


# ============================================================================
# Policy Store
# ============================================================================

def _compute_config_hash(config: PolicyConfig) -> str:
    """Compute deterministic hash of configuration for comparison."""
    config_str = json.dumps(config.model_dump(), sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def _generate_policy_id(option_id: str, config_hash: str) -> str:
    """Generate unique policy ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return f"pol_{ts}_{config_hash[:8]}"


class PolicyStore:
    """Thread-safe store for policy decisions.

    Stores policies in-memory for fast access and persists to JSONL for durability.
    Similar pattern to TelemetryStore for consistency.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self._policies: List[PolicyDecision] = []
        self._lock = threading.Lock()
        self._storage_dir = storage_dir or Path(os.getenv("ADS_POLICY_DIR", "./data/policies"))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._storage_file = self._storage_dir / "decisions.jsonl"

        # Load existing policies on startup
        self._load_from_file()

    def _load_from_file(self) -> None:
        """Load existing policies from storage file."""
        if not self._storage_file.exists():
            return

        try:
            with open(self._storage_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            policy = PolicyDecision.model_validate(data)
                            self._policies.append(policy)
                        except Exception:
                            # Skip malformed lines
                            pass
        except Exception:
            pass  # File might not exist or be empty

    def record(self, request: PolicySelectRequest) -> PolicyDecision:
        """Record a new policy decision."""
        config_hash = _compute_config_hash(request.config)
        policy_id = _generate_policy_id(request.option_id, config_hash)

        policy = PolicyDecision(
            policy_id=policy_id,
            session_id=request.session_id,
            option_id=request.option_id,
            option_text=request.option_text,
            objective_scores=request.objective_scores,
            is_pareto=request.is_pareto,
            config=request.config,
            config_hash=config_hash,
            rationale=request.rationale,
            metadata=request.metadata,
        )

        with self._lock:
            self._policies.append(policy)

            # Persist to file
            with open(self._storage_file, "a", encoding="utf-8") as f:
                policy_dict = policy.model_dump()
                # Convert datetime to ISO string
                policy_dict["created_at"] = policy_dict["created_at"].isoformat()
                f.write(json.dumps(policy_dict) + "\n")

        return policy

    def get_history(
        self,
        session_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        option_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[PolicyDecision], int]:
        """Query policy history with filtering.

        Returns (filtered_policies, total_count_before_pagination).
        """
        with self._lock:
            policies = self._policies.copy()

        # Apply filters
        if session_id:
            policies = [p for p in policies if p.session_id == session_id]
        if dataset_id:
            policies = [p for p in policies if p.config.dataset_id == dataset_id]
        if option_id:
            policies = [p for p in policies if p.option_id == option_id]

        filtered_count = len(policies)

        # Sort by created_at descending (most recent first)
        policies.sort(key=lambda p: p.created_at, reverse=True)

        # Apply pagination
        policies = policies[offset:offset + limit]

        return policies, filtered_count

    def get_by_id(self, policy_id: str) -> Optional[PolicyDecision]:
        """Get a specific policy by ID."""
        with self._lock:
            for p in self._policies:
                if p.policy_id == policy_id:
                    return p
        return None

    def get_stats(self) -> PolicyStatsResponse:
        """Compute aggregate statistics."""
        with self._lock:
            policies = self._policies.copy()

        if not policies:
            return PolicyStatsResponse(
                total_decisions=0,
                unique_sessions=0,
                decisions_by_dataset={},
                decisions_by_option={},
                most_selected_options=[],
                date_range=None,
            )

        sessions = set()
        by_dataset: Dict[str, int] = {}
        by_option: Dict[str, int] = {}

        for p in policies:
            if p.session_id:
                sessions.add(p.session_id)
            ds = p.config.dataset_id
            by_dataset[ds] = by_dataset.get(ds, 0) + 1
            by_option[p.option_id] = by_option.get(p.option_id, 0) + 1

        # Most selected options
        sorted_options = sorted(by_option.items(), key=lambda x: x[1], reverse=True)
        most_selected = [
            {"option_id": oid, "count": count}
            for oid, count in sorted_options[:10]
        ]

        # Date range
        timestamps = [p.created_at for p in policies]
        date_range = {
            "earliest": min(timestamps).isoformat(),
            "latest": max(timestamps).isoformat(),
        }

        return PolicyStatsResponse(
            total_decisions=len(policies),
            unique_sessions=len(sessions),
            decisions_by_dataset=by_dataset,
            decisions_by_option=by_option,
            most_selected_options=most_selected,
            date_range=date_range,
        )

    def clear(self) -> int:
        """Clear all policies (in-memory only, file persists).

        Returns count of cleared policies.
        """
        with self._lock:
            count = len(self._policies)
            self._policies = []
        return count


# Global store instance
_store = PolicyStore()


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/select", response_model=PolicySelectResponse)
def select_policy(request: PolicySelectRequest):
    """Record a policy selection.

    Call this when a user selects a Pareto-optimal option as their decision.
    The selection is persisted with full configuration context for audit.

    Example:
        POST /policy/select
        {
            "session_id": "sess_abc123",
            "option_id": "course_cs101",
            "option_text": "Introduction to Computer Science",
            "objective_scores": {"market": 0.85, "mission": 0.72, "university": 0.90},
            "is_pareto": true,
            "config": {
                "dataset_id": "paper_v1",
                "embedding_kind": "sbert",
                "objectives": ["market", "mission", "university"],
                "lens_mode": "learned",
                "autonomy_tau": 0.25
            },
            "rationale": "Best balance of market relevance and mission alignment"
        }
    """
    policy = _store.record(request)
    return PolicySelectResponse(
        ok=True,
        policy=policy,
        message=f"Policy {policy.policy_id} recorded successfully",
    )


class PolicyHistoryRequest(BaseModel):
    """Request parameters for policy history."""
    session_id: Optional[str] = None
    dataset_id: Optional[str] = None
    option_id: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


@router.post("/history", response_model=PolicyHistoryResponse)
def get_policy_history(request: PolicyHistoryRequest):
    """Get policy decision history with optional filtering.

    Filter by:
    - session_id: Decisions from a specific session
    - dataset_id: Decisions using a specific dataset
    - option_id: Decisions selecting a specific option

    Results are paginated (limit/offset) and sorted by most recent first.
    """
    policies, filtered_count = _store.get_history(
        session_id=request.session_id,
        dataset_id=request.dataset_id,
        option_id=request.option_id,
        limit=request.limit,
        offset=request.offset,
    )

    return PolicyHistoryResponse(
        policies=policies,
        total_count=filtered_count,
        filtered_count=len(policies),
    )


@router.get("/history", response_model=PolicyHistoryResponse)
def get_policy_history_get(
    session_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    limit: int = 100,
):
    """Get policy history (GET version for simpler queries)."""
    policies, filtered_count = _store.get_history(
        session_id=session_id,
        dataset_id=dataset_id,
        limit=min(limit, 100),
    )

    return PolicyHistoryResponse(
        policies=policies,
        total_count=filtered_count,
        filtered_count=len(policies),
    )


@router.get("/{policy_id}", response_model=PolicyDecision)
def get_policy(policy_id: str):
    """Get a specific policy decision by ID."""
    policy = _store.get_by_id(policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail=f"Policy not found: {policy_id}")
    return policy


@router.get("/stats/summary", response_model=PolicyStatsResponse)
def get_policy_stats():
    """Get aggregate statistics about policy decisions.

    Returns:
    - Total decision count
    - Unique session count
    - Decisions by dataset
    - Most frequently selected options
    - Date range of decisions
    """
    return _store.get_stats()


@router.delete("/clear")
def clear_policies():
    """Clear in-memory policy cache.

    Note: The persistent file is NOT cleared. Use this for testing/development.
    """
    count = _store.clear()
    return {"ok": True, "cleared": count, "message": "In-memory policies cleared"}
