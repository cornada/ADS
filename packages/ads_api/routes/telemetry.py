"""Telemetry logging for KT8.

Logs user interactions for research analysis:
- Session tracking
- Toggle interactions (objective enable/disable, tau changes)
- Option selections and explanations viewed
- Time spent on views

Events are logged to both memory (for real-time queries) and file (for persistence).
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import threading

router = APIRouter()


# Event types for the dashboard
class EventType:
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CONFIG_CHANGE = "config_change"
    OBJECTIVE_TOGGLE = "objective_toggle"
    TAU_CHANGE = "tau_change"
    LENS_CHANGE = "lens_change"
    OPTION_SELECT = "option_select"
    OPTION_HOVER = "option_hover"
    EXPLAIN_VIEW = "explain_view"
    RECOURSE_VIEW = "recourse_view"
    PARETO_REFRESH = "pareto_refresh"


class TelemetryEvent(BaseModel):
    """A telemetry event from the dashboard."""

    session_id: str = Field(..., description="Unique session identifier")
    event_type: str = Field(..., description="Type of event")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event-specific data")
    ts: Optional[datetime] = Field(default=None, description="Event timestamp")
    user_agent: Optional[str] = Field(default=None, description="Browser user agent")

    def with_timestamp(self) -> "TelemetryEvent":
        """Ensure timestamp is set."""
        if self.ts is None:
            return TelemetryEvent(
                session_id=self.session_id,
                event_type=self.event_type,
                payload=self.payload,
                ts=datetime.now(timezone.utc),
                user_agent=self.user_agent,
            )
        return self


class TelemetryStore:
    """Thread-safe store for telemetry events."""

    def __init__(self, log_dir: Optional[Path] = None):
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._log_dir = log_dir or Path(os.getenv("ADS_TELEMETRY_DIR", "./data/telemetry"))
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "events.jsonl"

    def log(self, event: TelemetryEvent) -> Dict[str, Any]:
        """Log an event to memory and file."""
        event = event.with_timestamp()
        event_dict = event.model_dump()

        # Convert datetime to ISO string for JSON
        if event_dict.get("ts"):
            event_dict["ts"] = event_dict["ts"].isoformat()

        with self._lock:
            self._events.append(event_dict)

            # Append to file
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_dict) + "\n")

        return event_dict

    def get_events(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query events from memory."""
        with self._lock:
            events = self._events.copy()

        # Filter
        if session_id:
            events = [e for e in events if e.get("session_id") == session_id]
        if event_type:
            events = [e for e in events if e.get("event_type") == event_type]

        # Return most recent
        return events[-limit:]

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary statistics for a session."""
        events = self.get_events(session_id=session_id, limit=1000)

        if not events:
            return {"session_id": session_id, "event_count": 0}

        # Compute stats
        event_types = {}
        for e in events:
            t = e.get("event_type", "unknown")
            event_types[t] = event_types.get(t, 0) + 1

        # Find time range
        timestamps = [e.get("ts") for e in events if e.get("ts")]
        if timestamps:
            start = min(timestamps)
            end = max(timestamps)
        else:
            start = end = None

        return {
            "session_id": session_id,
            "event_count": len(events),
            "event_types": event_types,
            "start_time": start,
            "end_time": end,
        }

    def clear(self):
        """Clear in-memory events (file persists)."""
        with self._lock:
            self._events = []


# Global store instance
_store = TelemetryStore()


@router.post("/event")
def log_event(evt: TelemetryEvent):
    """Log a telemetry event.

    Returns the logged event with timestamp.
    """
    logged = _store.log(evt)
    return {"ok": True, "event": logged}


class BatchEventsRequest(BaseModel):
    """Request to log multiple events."""

    events: List[TelemetryEvent]


@router.post("/events")
def log_events(req: BatchEventsRequest):
    """Log multiple events in batch."""
    logged = []
    for evt in req.events:
        logged.append(_store.log(evt))
    return {"ok": True, "count": len(logged), "events": logged}


class QueryEventsRequest(BaseModel):
    """Request to query events."""

    session_id: Optional[str] = None
    event_type: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)


@router.post("/query")
def query_events(req: QueryEventsRequest):
    """Query logged events."""
    events = _store.get_events(
        session_id=req.session_id,
        event_type=req.event_type,
        limit=req.limit,
    )
    return {"events": events, "count": len(events)}


@router.get("/session/{session_id}")
def get_session_summary(session_id: str):
    """Get summary for a session."""
    return _store.get_session_summary(session_id)


@router.get("/stats")
def get_stats():
    """Get overall telemetry stats."""
    all_events = _store.get_events(limit=10000)

    sessions = set()
    event_types = {}
    for e in all_events:
        sessions.add(e.get("session_id"))
        t = e.get("event_type", "unknown")
        event_types[t] = event_types.get(t, 0) + 1

    return {
        "total_events": len(all_events),
        "unique_sessions": len(sessions),
        "event_types": event_types,
        "log_file": str(_store._log_file),
    }
