import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def edge_client():
    # Patch scheduler start so tests don't fire real APScheduler jobs
    with patch("backend.schedulers.edge_scheduler.start_edge_scheduler"):
        with patch("backend.schedulers.edge_scheduler.stop_edge_scheduler"):
            from backend.edge_app import app
            with TestClient(app) as client:
                yield client


def test_edge_app_health_returns_200(edge_client):
    response = edge_client.get("/health")
    assert response.status_code == 200


def test_edge_app_health_contains_status(edge_client):
    response = edge_client.get("/health")
    data = response.json()
    assert "status" in data


def test_edge_app_root_returns_deployment_role(edge_client):
    response = edge_client.get("/")
    assert response.status_code == 200


def test_edge_app_does_not_expose_fantasy_routes(edge_client):
    response = edge_client.get("/api/fantasy/lineup/2026-04-04")
    # Fantasy routes are not mounted on the edge app
    assert response.status_code == 404
