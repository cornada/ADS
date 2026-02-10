"""Tests for Phase 4 API routes."""
import pytest
from fastapi.testclient import TestClient

from ads_api.main import app

client = TestClient(app)


# ============================================================================
# Health + Index
# ============================================================================

class TestHealth:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_index(self):
        r = client.get("/")
        assert r.status_code == 200


# ============================================================================
# Courses
# ============================================================================

class TestCourses:
    def test_list_empty(self):
        r = client.get("/courses/")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_not_found(self):
        r = client.get("/courses/nonexistent")
        assert r.status_code == 404

    def test_search_empty(self):
        r = client.post("/courses/search", json={"query": "python"})
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_stats(self):
        r = client.get("/courses/stats/summary")
        assert r.status_code == 200
        assert "total_courses" in r.json()


# ============================================================================
# Learner
# ============================================================================

class TestLearner:
    def test_register_and_get(self):
        r = client.post("/learner/register?learner_id=test_student_1")
        assert r.status_code == 200
        data = r.json()
        assert data["learner_id"] == "test_student_1"
        assert "competencies" in data

        r2 = client.get("/learner/test_student_1")
        assert r2.status_code == 200
        assert r2.json()["learner_id"] == "test_student_1"

    def test_register_duplicate(self):
        client.post("/learner/register?learner_id=dup_student")
        r = client.post("/learner/register?learner_id=dup_student")
        assert r.status_code == 409

    def test_get_not_found(self):
        r = client.get("/learner/nonexistent_student")
        assert r.status_code == 404

    def test_observe(self):
        client.post("/learner/register?learner_id=obs_student")
        r = client.post("/learner/observe", json={
            "learner_id": "obs_student",
            "scores": [0.7, 0.5, 0.6, 0.4],  # 4 observation dims (3 comp + engagement)
        })
        assert r.status_code == 200
        assert r.json()["timestep"] == 1

    def test_observe_wrong_dim(self):
        client.post("/learner/register?learner_id=dim_student")
        r = client.post("/learner/observe", json={
            "learner_id": "dim_student",
            "scores": [0.5],  # wrong dimension (need 4)
        })
        assert r.status_code == 400

    def test_recommend(self):
        client.post("/learner/register?learner_id=rec_student")
        r = client.post("/learner/recommend", json={
            "learner_id": "rec_student",
            "policy": "greedy",
            "n_recommendations": 3,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["recommendations"]) == 3
        assert data["profile"]["learner_id"] == "rec_student"

    def test_list_active(self):
        r = client.get("/learner/list/active")
        assert r.status_code == 200
        assert r.json()["active_learners"] >= 0


# ============================================================================
# Assessment (CAT)
# ============================================================================

class TestAssessment:
    def test_start_session(self):
        r = client.post("/assessment/start", json={
            "learner_id": "cat_student_1",
            "max_items": 10,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["item"] is not None
        assert data["session"]["n_administered"] == 0
        assert not data["session"]["terminated"]

    def test_respond_and_progress(self):
        # Start session
        r = client.post("/assessment/start", json={
            "learner_id": "cat_student_2",
            "max_items": 5,
        })
        data = r.json()
        session_id = data["session"]["session_id"]
        item_id = data["item"]["item_id"]

        # Respond to first item
        r2 = client.post("/assessment/respond", json={
            "session_id": session_id,
            "item_id": item_id,
            "correct": True,
        })
        assert r2.status_code == 200
        data2 = r2.json()
        assert data2["session"]["n_administered"] == 1

    def test_full_cat_session(self):
        r = client.post("/assessment/start", json={
            "learner_id": "cat_student_full",
            "max_items": 5,
        })
        data = r.json()
        session_id = data["session"]["session_id"]

        # Answer all items
        for i in range(5):
            item = data.get("item")
            if item is None:
                break
            r = client.post("/assessment/respond", json={
                "session_id": session_id,
                "item_id": item["item_id"],
                "correct": i % 2 == 0,
            })
            data = r.json()

        # Should be terminated
        assert data["session"]["terminated"]

    def test_session_not_found(self):
        r = client.post("/assessment/respond", json={
            "session_id": "nonexistent",
            "item_id": "item_000",
            "correct": True,
        })
        assert r.status_code == 404

    def test_session_history(self):
        r = client.post("/assessment/start", json={
            "learner_id": "cat_hist",
            "max_items": 10,
        })
        data = r.json()
        assert "session" in data, f"Unexpected response: {data}"
        session_id = data["session"]["session_id"]
        item_id = data["item"]["item_id"]

        client.post("/assessment/respond", json={
            "session_id": session_id,
            "item_id": item_id,
            "correct": True,
        })

        r2 = client.get(f"/assessment/{session_id}/history")
        assert r2.status_code == 200
        assert len(r2.json()["responses"]) == 1


# ============================================================================
# Temporal
# ============================================================================

class TestTemporal:
    def test_overview_empty(self):
        r = client.get("/temporal/overview")
        assert r.status_code == 200
        assert "years_available" in r.json()

    def test_changepoints_empty(self):
        r = client.get("/temporal/changepoints")
        assert r.status_code == 200


# ============================================================================
# Transport
# ============================================================================

class TestTransport:
    def test_alignment_not_found(self):
        r = client.get("/transport/alignment/nonexistent")
        assert r.status_code == 404


# ============================================================================
# Policy (existing)
# ============================================================================

class TestPolicy:
    def test_stats(self):
        r = client.get("/policy/stats/summary")
        assert r.status_code == 200
        assert "total_decisions" in r.json()
