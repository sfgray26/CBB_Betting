"""Tests for admin endpoint registration and compatibility contracts."""

import pytest

import backend.main as main


def test_admin_version_endpoint_exists():
    """GET /admin/version should exist and be registered."""
    # Verify the route is registered
    routes = [route.path for route in main.app.routes]
    assert "/admin/version" in routes, "/admin/version endpoint should be registered"
    assert "/admin/audit-tables" in routes, "/admin/audit-tables compatibility route should be registered"


def test_deployment_version_model_exists():
    """DeploymentVersion model should exist with required fields."""
    from backend.models import DeploymentVersion

    # Verify the model exists with required fields
    assert hasattr(DeploymentVersion, 'git_commit_sha')
    assert hasattr(DeploymentVersion, 'git_commit_date')
    assert hasattr(DeploymentVersion, 'build_timestamp')
    assert hasattr(DeploymentVersion, 'app_version')


@pytest.mark.asyncio
async def test_admin_yahoo_test_includes_connected(monkeypatch):
    """Yahoo connectivity endpoint should expose a stable connected flag."""
    class FakeYahooClient:
        league_key = "mlb.l.12345"

        def get_league(self):
            return {"name": "Test League"}

        def get_my_team_key(self):
            return "mlb.l.12345.t.1"

    monkeypatch.setattr(main, "get_yahoo_client", lambda: FakeYahooClient())

    data = await main.yahoo_test(user="test-admin")

    assert data["connected"] is True
    assert data["league_name"] == "Test League"


@pytest.mark.asyncio
async def test_admin_odds_monitor_status_degrades_when_client_missing(monkeypatch):
    """Odds monitor status should not 500 when odds configuration is missing."""
    def _raise_missing_client():
        raise ValueError("THE_ODDS_API_KEY not set")

    monkeypatch.setattr(main, "get_odds_monitor", _raise_missing_client)

    data = await main.get_odds_monitor_status(user="test-user")

    assert data["status"] == "degraded"
    assert data["active"] is False
    assert "THE_ODDS_API_KEY" in data["error"]
