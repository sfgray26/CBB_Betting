import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture
def fantasy_client():
    with patch("backend.schedulers.fantasy_scheduler.start_fantasy_scheduler"):
        with patch("backend.schedulers.fantasy_scheduler.stop_fantasy_scheduler"):
            from backend.fantasy_app import app
            with TestClient(app) as client:
                yield client


def test_fantasy_app_health_returns_200(fantasy_client):
    response = fantasy_client.get("/health")
    assert response.status_code == 200


def test_fantasy_app_health_contains_status(fantasy_client):
    response = fantasy_client.get("/health")
    data = response.json()
    assert "status" in data


def test_fantasy_app_does_not_expose_edge_routes(fantasy_client):
    response = fantasy_client.get("/api/predictions/today")
    # Edge routes are not mounted on the fantasy app
    assert response.status_code == 404


def test_fantasy_app_root_returns_200(fantasy_client):
    response = fantasy_client.get("/")
    assert response.status_code == 200
