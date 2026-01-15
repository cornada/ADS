"""Tests for dashboard and telemetry API endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ads_api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health(self, client):
        """Test health endpoint returns ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestDashboardEndpoints:
    """Tests for dashboard API endpoints."""

    def test_get_dashboard_data_default_config(self, client):
        """Test getting dashboard data with default config."""
        response = client.post("/dashboard/data", json={})

        assert response.status_code == 200
        data = response.json()

        assert "config" in data
        assert "options" in data
        assert "pareto_ids" in data
        assert "objective_keys" in data
        assert "stats" in data

        # Check stats
        assert data["stats"]["total_options"] > 0
        assert data["stats"]["pareto_count"] > 0

    def test_get_dashboard_data_custom_objectives(self, client):
        """Test dashboard with custom objectives."""
        response = client.post("/dashboard/data", json={
            "objectives": ["market", "learner"],
            "autonomy_tau": 0.5,
        })

        assert response.status_code == 200
        data = response.json()

        assert set(data["objective_keys"]) == {"market", "learner"}
        assert data["config"]["autonomy_tau"] == 0.5

    def test_get_dashboard_data_different_tau(self, client):
        """Test that different tau values affect feasibility."""
        # Strict tau
        response1 = client.post("/dashboard/data", json={
            "autonomy_tau": 0.1,
        })
        data1 = response1.json()

        # Relaxed tau
        response2 = client.post("/dashboard/data", json={
            "autonomy_tau": 0.9,
        })
        data2 = response2.json()

        # More options should be feasible with relaxed tau
        assert data2["stats"]["feasible_count"] >= data1["stats"]["feasible_count"]

    def test_explain_option(self, client):
        """Test explaining an option."""
        # First get available options
        data_response = client.post("/dashboard/data", json={})
        data = data_response.json()
        option_id = data["options"][0]["option_id"]

        # Request explanation
        response = client.post("/dashboard/explain", json={
            "option_id": option_id,
            "config": {}
        })

        assert response.status_code == 200
        exp_data = response.json()

        assert exp_data["option_id"] == option_id
        assert "explanation" in exp_data
        assert "overall_summary" in exp_data["explanation"]
        assert "pareto_status" in exp_data["explanation"]

    def test_explain_unknown_option(self, client):
        """Test explaining an unknown option returns 404."""
        response = client.post("/dashboard/explain", json={
            "option_id": "course:nonexistent",
            "config": {}
        })

        assert response.status_code == 404


class TestTelemetryEndpoints:
    """Tests for telemetry API endpoints."""

    def test_log_single_event(self, client):
        """Test logging a single event."""
        response = client.post("/telemetry/event", json={
            "session_id": "test_session_123",
            "event_type": "test_event",
            "payload": {"key": "value"}
        })

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "event" in data
        assert data["event"]["session_id"] == "test_session_123"
        assert data["event"]["event_type"] == "test_event"
        assert "ts" in data["event"]

    def test_log_batch_events(self, client):
        """Test logging multiple events."""
        response = client.post("/telemetry/events", json={
            "events": [
                {"session_id": "batch_session", "event_type": "event1"},
                {"session_id": "batch_session", "event_type": "event2"},
                {"session_id": "batch_session", "event_type": "event3"},
            ]
        })

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["count"] == 3

    def test_query_events(self, client):
        """Test querying events."""
        # Log some events first
        client.post("/telemetry/event", json={
            "session_id": "query_test",
            "event_type": "query_event_1"
        })
        client.post("/telemetry/event", json={
            "session_id": "query_test",
            "event_type": "query_event_2"
        })

        # Query by session
        response = client.post("/telemetry/query", json={
            "session_id": "query_test",
            "limit": 10
        })

        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 2
        for event in data["events"]:
            assert event["session_id"] == "query_test"

    def test_session_summary(self, client):
        """Test getting session summary."""
        session_id = "summary_test_session"

        # Log some events
        client.post("/telemetry/event", json={
            "session_id": session_id,
            "event_type": "session_start"
        })
        client.post("/telemetry/event", json={
            "session_id": session_id,
            "event_type": "config_change"
        })

        response = client.get(f"/telemetry/session/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["event_count"] >= 2

    def test_telemetry_stats(self, client):
        """Test getting overall telemetry stats."""
        response = client.get("/telemetry/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_events" in data
        assert "unique_sessions" in data
        assert "event_types" in data


class TestOptionDetails:
    """Tests for option detail information."""

    def test_options_have_required_fields(self, client):
        """Test that options have all required fields."""
        response = client.post("/dashboard/data", json={})
        data = response.json()

        for option in data["options"]:
            assert "option_id" in option
            assert "text" in option
            assert "objectives" in option
            assert "feasible" in option
            assert "is_pareto" in option
            assert "autonomy_drift" in option

    def test_pareto_options_marked(self, client):
        """Test that Pareto options are properly marked."""
        response = client.post("/dashboard/data", json={})
        data = response.json()

        pareto_ids = set(data["pareto_ids"])
        for option in data["options"]:
            if option["option_id"] in pareto_ids:
                assert option["is_pareto"] is True
            else:
                assert option["is_pareto"] is False
