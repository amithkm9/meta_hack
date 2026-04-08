"""Tests for the FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHealth:
    def test_health(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestTasks:
    def test_list_tasks(self, client: TestClient):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 3
        ids = {t["id"] for t in data}
        assert "easy_remediate_handshape" in ids
        assert "medium_fix_movement_with_scaffold" in ids
        assert "hard_adaptive_multi_error_plan" in ids


class TestReset:
    def test_reset_valid_task(self, client: TestClient):
        resp = client.post("/reset", json={"task_id": "easy_remediate_handshape"})
        assert resp.status_code == 200
        data = resp.json()
        assert "episode_id" in data
        assert data["done"] is False
        assert data["observation"]["task_id"] == "easy_remediate_handshape"

    def test_reset_invalid_task(self, client: TestClient):
        resp = client.post("/reset", json={"task_id": "nonexistent"})
        assert resp.status_code == 404


class TestStep:
    def test_step_valid(self, client: TestClient):
        reset_resp = client.post("/reset", json={"task_id": "easy_remediate_handshape"})
        episode_id = reset_resp.json()["episode_id"]

        step_resp = client.post("/step", json={
            "episode_id": episode_id,
            "action": {
                "action_type": "slow_motion_demo",
                "rationale": "demo",
                "payload": {},
            },
        })
        assert step_resp.status_code == 200
        data = step_resp.json()
        assert "reward" in data
        assert "done" in data

    def test_step_wrong_episode(self, client: TestClient):
        client.post("/reset", json={"task_id": "easy_remediate_handshape"})
        step_resp = client.post("/step", json={
            "episode_id": "wrong_id",
            "action": {"action_type": "slow_motion_demo", "rationale": "", "payload": {}},
        })
        assert step_resp.status_code == 400


class TestState:
    def test_state_after_reset(self, client: TestClient):
        client.post("/reset", json={"task_id": "easy_remediate_handshape"})
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "easy_remediate_handshape"
        assert data["step_count"] == 0
        assert data["done"] is False
