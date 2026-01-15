#!/usr/bin/env python
"""Demo script for KT8: Dashboard API and Telemetry.

This script demonstrates:
1. Starting the API server
2. Fetching Pareto data with different configurations
3. Logging telemetry events
4. Viewing telemetry stats

Usage:
    # Start the server first:
    uvicorn ads_api.main:app --reload --port 8000

    # Then run this demo:
    python scripts/demo_dashboard.py
"""
from __future__ import annotations

import requests
import uuid

API_BASE = "http://localhost:8000"


def check_server():
    """Check if the server is running."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def demo_dashboard_data():
    """Demo: Fetch dashboard data with different configurations."""
    print("=" * 70)
    print("DEMO: Dashboard Data API")
    print("=" * 70)

    # Default configuration
    print("\n[1] Fetching with default config...")
    r = requests.post(f"{API_BASE}/dashboard/data", json={})
    data = r.json()

    print(f"  Total options: {data['stats']['total_options']}")
    print(f"  Pareto optimal: {data['stats']['pareto_count']}")
    print(f"  Feasible: {data['stats']['feasible_count']}")
    print(f"  Objectives: {data['objective_keys']}")

    print("\n  Pareto options:")
    for opt in data["options"]:
        if opt["is_pareto"]:
            scores = ", ".join(f"{k}={v:.3f}" for k, v in opt["objectives"].items())
            print(f"    - {opt['option_id']}: {scores}")

    # Strict tau
    print("\n[2] Fetching with strict tau=0.1...")
    r = requests.post(f"{API_BASE}/dashboard/data", json={
        "autonomy_tau": 0.1,
    })
    data_strict = r.json()
    print(f"  Feasible with tau=0.1: {data_strict['stats']['feasible_count']}")

    # Relaxed tau
    print("\n[3] Fetching with relaxed tau=0.5...")
    r = requests.post(f"{API_BASE}/dashboard/data", json={
        "autonomy_tau": 0.5,
    })
    data_relaxed = r.json()
    print(f"  Feasible with tau=0.5: {data_relaxed['stats']['feasible_count']}")

    # Only market + learner
    print("\n[4] Fetching with only market + learner objectives...")
    r = requests.post(f"{API_BASE}/dashboard/data", json={
        "objectives": ["market", "learner"],
    })
    data_subset = r.json()
    print(f"  Objectives: {data_subset['objective_keys']}")
    print(f"  Pareto count: {data_subset['stats']['pareto_count']}")

    return data


def demo_explanation(option_id: str):
    """Demo: Get explanation for an option."""
    print("\n" + "=" * 70)
    print(f"DEMO: Explanation for {option_id}")
    print("=" * 70)

    r = requests.post(f"{API_BASE}/dashboard/explain", json={
        "option_id": option_id,
        "config": {},
    })

    if r.status_code != 200:
        print(f"  Error: {r.json()}")
        return

    data = r.json()
    exp = data["explanation"]

    print(f"\n  Summary: {exp['overall_summary']}")
    print(f"  Status: {exp['pareto_status'].upper()}")
    print(f"  Reason: {exp['pareto_reason']}")

    if exp.get("drift"):
        print("\n  Autonomy Drift:")
        print(f"    Total: {exp['drift']['total_drift']:.4f}")
        print(f"    Interpretation: {exp['drift']['interpretation']}")

    if data.get("recourse"):
        rec = data["recourse"]
        if rec.get("alternatives"):
            print("\n  Recourse Suggestion:")
            print(f"    {rec['summary']}")
            best = rec.get("best_alternative")
            if best:
                print(f"    Best: {best['option_id']} (+{best['improvement']:.3f} on {best['target_objective']})")


def demo_telemetry():
    """Demo: Telemetry logging and querying."""
    print("\n" + "=" * 70)
    print("DEMO: Telemetry System")
    print("=" * 70)

    session_id = f"demo_{uuid.uuid4().hex[:8]}"
    print(f"\n  Session ID: {session_id}")

    # Log session start
    print("\n[1] Logging session_start event...")
    r = requests.post(f"{API_BASE}/telemetry/event", json={
        "session_id": session_id,
        "event_type": "session_start",
        "payload": {"demo": True},
    })
    print(f"  Response: {r.json()['ok']}")

    # Log config change
    print("\n[2] Logging config_change event...")
    requests.post(f"{API_BASE}/telemetry/event", json={
        "session_id": session_id,
        "event_type": "config_change",
        "payload": {"tau": 0.25, "objectives": ["market", "learner"]},
    })

    # Log option selection
    print("\n[3] Logging option_select event...")
    requests.post(f"{API_BASE}/telemetry/event", json={
        "session_id": session_id,
        "event_type": "option_select",
        "payload": {"option_id": "course:dl"},
    })

    # Query session events
    print("\n[4] Querying session events...")
    r = requests.post(f"{API_BASE}/telemetry/query", json={
        "session_id": session_id,
        "limit": 10,
    })
    events = r.json()
    print(f"  Found {events['count']} events")
    for e in events["events"]:
        print(f"    - {e['event_type']}: {e.get('payload', {})}")

    # Get session summary
    print("\n[5] Getting session summary...")
    r = requests.get(f"{API_BASE}/telemetry/session/{session_id}")
    summary = r.json()
    print(f"  Event count: {summary['event_count']}")
    print(f"  Event types: {summary['event_types']}")

    # Get overall stats
    print("\n[6] Getting overall telemetry stats...")
    r = requests.get(f"{API_BASE}/telemetry/stats")
    stats = r.json()
    print(f"  Total events: {stats['total_events']}")
    print(f"  Unique sessions: {stats['unique_sessions']}")
    print(f"  Log file: {stats['log_file']}")


def main():
    print("=" * 70)
    print("KT8 DEMO: Dashboard API + Telemetry")
    print("=" * 70)

    # Check server
    if not check_server():
        print("\nError: API server not running!")
        print("Start it with: uvicorn ads_api.main:app --reload --port 8000")
        return

    print("\nServer is running at", API_BASE)

    # Run demos
    data = demo_dashboard_data()

    # Get a Pareto option for explanation
    pareto_options = [o for o in data["options"] if o["is_pareto"]]
    if pareto_options:
        demo_explanation(pareto_options[0]["option_id"])

    # Get a non-Pareto option for recourse
    non_pareto = [o for o in data["options"] if not o["is_pareto"]]
    if non_pareto:
        demo_explanation(non_pareto[0]["option_id"])

    demo_telemetry()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
KT8 Dashboard Features:
  1. Pareto visualization endpoint: /dashboard/data
  2. Toggle controls: objectives, tau, lens_mode
  3. Explanation endpoint: /dashboard/explain
  4. Telemetry logging: /telemetry/event
  5. Telemetry queries: /telemetry/query, /telemetry/session/{id}
  6. UI dashboard: http://localhost:8000/ or /static/index.html

To run the full UI:
  uvicorn ads_api.main:app --reload
  Open http://localhost:8000 in your browser
""")


if __name__ == "__main__":
    main()
