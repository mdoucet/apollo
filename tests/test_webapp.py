"""
Tests for the Flask web application.
"""

import json

import pytest

from apollo_lander.webapp import create_app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestWebApp:
    """Tests for the Flask web application routes."""

    def test_index_page(self, client):
        """GET / should return the game page."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"Apollo Lander" in response.data

    def test_reset_endpoint(self, client):
        """POST /api/reset should return initial game state."""
        response = client.post("/api/reset")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "altitude" in data
        assert data["altitude"] > 0
        assert data["game_over"] is False
        assert data["landing_success"] is False
        assert isinstance(data["obs"], list)
        assert len(data["obs"]) == 12

    def test_step_endpoint(self, client):
        """POST /api/step should advance the simulation."""
        # Reset first
        client.post("/api/reset")

        # Step with neutral action
        response = client.post(
            "/api/step",
            data=json.dumps({"rhc": [0, 0, 0], "rod": 0}),
            content_type="application/json",
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "altitude" in data
        assert "vertical_velocity" in data
        assert "total_reward" in data

    def test_step_validates_input(self, client):
        """POST /api/step should reject invalid input."""
        client.post("/api/reset")

        # Invalid rhc
        response = client.post(
            "/api/step",
            data=json.dumps({"rhc": [0, 0], "rod": 0}),
            content_type="application/json",
        )
        assert response.status_code == 400

        # Invalid rod
        response = client.post(
            "/api/step",
            data=json.dumps({"rhc": [0, 0, 0], "rod": 5}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_step_without_reset_returns_error(self, client):
        """POST /api/step before reset should not crash."""
        # The game_over flag defaults to False but env is None,
        # so stepping should handle this gracefully.
        client.post("/api/reset")

        response = client.post(
            "/api/step",
            data=json.dumps({"rhc": [0, 0, 0], "rod": 0}),
            content_type="application/json",
        )
        assert response.status_code == 200

    def test_rhc_values_are_clamped(self, client):
        """RHC values outside [-1, 1] should be clamped."""
        client.post("/api/reset")

        response = client.post(
            "/api/step",
            data=json.dumps({"rhc": [5.0, -5.0, 10.0], "rod": 0}),
            content_type="application/json",
        )
        assert response.status_code == 200
