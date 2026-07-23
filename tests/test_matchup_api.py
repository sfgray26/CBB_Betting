"""
Tests for Matchup API endpoint.

Tests for GET /api/fantasy/matchup after cache removal (Loop 9).
Verifies live data behavior and 504 timeout safety valve.
"""

import pytest
from unittest.mock import patch, MagicMock
import time


@pytest.fixture
def fantasy_client():
    with patch("backend.schedulers.fantasy_scheduler.start_fantasy_scheduler"):
        with patch("backend.schedulers.fantasy_scheduler.stop_fantasy_scheduler"):
            from backend.fantasy_app import app
            from backend.auth import verify_api_key
            from backend.models import get_db as get_db_dependency
            from fastapi.testclient import TestClient

            mock_db = MagicMock()

            def override_get_db():
                try:
                    yield mock_db
                finally:
                    pass

            app.dependency_overrides[get_db_dependency] = override_get_db
            app.dependency_overrides[verify_api_key] = lambda: "test-user"
            with TestClient(app) as client:
                yield client
            app.dependency_overrides.clear()


class TestMatchupEndpointLiveData:
    """Tests for GET /api/fantasy/matchup live data behavior (no cache)."""

    @patch("backend.routers.fantasy.get_yahoo_client")
    def test_matchup_fetches_live_data_on_each_request(self, mock_client_factory, fantasy_client):
        """Each request should call Yahoo API — no cache means live data."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        mock_client = MagicMock(spec=YahooFantasyClient)
        mock_client.get_my_team_key.return_value = "388.l.123456.t.1"
        mock_client.get_scoreboard.return_value = [
            {
                "week": 12,
                "teams": [
                    {
                        "team": {
                            "team_key": "388.l.123456.t.1",
                            "name": "My Team",
                            "team_stats": {"stats": [{"stat": {"stat_id": "50", "value": "7"}}]},
                        }
                    },
                    {
                        "team": {
                            "team_key": "388.l.123456.t.2",
                            "name": "Opponent",
                            "team_stats": {"stats": [{"stat": {"stat_id": "50", "value": "5"}}]},
                        }
                    },
                ]
            }
        ]
        mock_client_factory.return_value = mock_client

        # First request
        response1 = fantasy_client.get("/api/fantasy/matchup", headers={"X-API-Key": "test-key"})
        assert response1.status_code == 200
        assert mock_client.get_scoreboard.call_count == 1

        # Second request — should call Yahoo again (no cache)
        response2 = fantasy_client.get("/api/fantasy/matchup", headers={"X-API-Key": "test-key"})
        assert response2.status_code == 200
        assert mock_client.get_scoreboard.call_count == 2

    @patch("backend.routers.fantasy.get_yahoo_client")
    def test_matchup_timeout_returns_504(self, mock_client_factory, fantasy_client):
        """Timeout (>5s) on Yahoo API call returns 504 Gateway Timeout."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        mock_client = MagicMock(spec=YahooFantasyClient)
        mock_client.get_my_team_key.return_value = "388.l.123456.t.1"

        # Simulate slow Yahoo call that exceeds 5-second timeout
        def slow_scoreboard():
            time.sleep(6)  # Exceeds 5-second timeout
            return []

        mock_client.get_scoreboard.side_effect = slow_scoreboard
        mock_client_factory.return_value = mock_client

        response = fantasy_client.get("/api/fantasy/matchup", headers={"X-API-Key": "test-key"})
        assert response.status_code == 504
        assert "timeout" in response.json()["detail"].lower()

    @patch("backend.routers.fantasy.get_yahoo_client")
    def test_matchup_response_structure_unchanged_after_cache_removal(self, mock_client_factory, fantasy_client):
        """Response shape remains identical — cache removal only affects freshness."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        mock_client = MagicMock(spec=YahooFantasyClient)
        mock_client.get_my_team_key.return_value = "388.l.123456.t.1"
        mock_client.get_scoreboard.return_value = [
            {
                "week": 12,
                "teams": [
                    {
                        "team": {
                            "team_key": "388.l.123456.t.1",
                            "name": "My Team",
                            "team_stats": {"stats": [{"stat": {"stat_id": "50", "value": "7"}}]},
                        }
                    },
                    {
                        "team": {
                            "team_key": "388.l.123456.t.2",
                            "name": "Opponent",
                            "team_stats": {"stats": [{"stat": {"stat_id": "50", "value": "5"}}]},
                        }
                    },
                ]
            }
        ]
        mock_client_factory.return_value = mock_client

        response = fantasy_client.get("/api/fantasy/matchup", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200

        data = response.json()
        assert "week" in data
        assert "my_team" in data
        assert "opponent" in data
        assert "is_playoffs" in data

        # Verify team structure
        assert "team_key" in data["my_team"]
        assert "team_name" in data["my_team"]
        assert "stats" in data["my_team"]
        assert isinstance(data["my_team"]["stats"], dict)

    @patch("backend.routers.fantasy.get_yahoo_client")
    def test_matchup_yahoo_auth_error_returns_still_works(self, mock_client_factory, fantasy_client):
        """Yahoo auth errors still handled correctly after cache removal."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
        from backend.fantasy_baseball.yahoo_client_resilient import YahooAuthError

        mock_client = MagicMock(spec=YahooFantasyClient)
        mock_client.get_my_team_key.return_value = "388.l.123456.t.1"
        mock_client.get_scoreboard.side_effect = YahooAuthError("Token expired")
        mock_client_factory.return_value = mock_client

        response = fantasy_client.get("/api/fantasy/matchup", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "Yahoo API error" in data["message"]
        # §R2: stub responses must be flagged degraded so the UI shows an honest
        # error/retry state instead of an all-tied 0-0 matchup.
        assert data["degraded"] is True

    @patch("backend.routers.fantasy.get_yahoo_client")
    def test_matchup_no_data_returns_message(self, mock_client_factory, fantasy_client):
        """Empty scoreboard returns appropriate message response."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        mock_client = MagicMock(spec=YahooFantasyClient)
        mock_client.get_my_team_key.return_value = "388.l.123456.t.1"
        mock_client.get_scoreboard.return_value = []
        mock_client_factory.return_value = mock_client

        response = fantasy_client.get("/api/fantasy/matchup", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "season may be starting" in data["message"]
        assert data["degraded"] is True

    @patch("backend.routers.fantasy.get_yahoo_client")
    def test_matchup_team_not_found_is_degraded(self, mock_client_factory, fantasy_client):
        """My team missing from the scoreboard returns a degraded stub (§R2)."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        mock_client = MagicMock(spec=YahooFantasyClient)
        mock_client.get_my_team_key.return_value = "388.l.123456.t.1"
        # Scoreboard has a matchup but NOT my team key → falls through to stub.
        mock_client.get_scoreboard.return_value = [
            {
                "week": 12,
                "teams": [
                    {"team": {"team_key": "388.l.123456.t.8", "name": "Other A",
                              "team_stats": {"stats": []}}},
                    {"team": {"team_key": "388.l.123456.t.9", "name": "Other B",
                              "team_stats": {"stats": []}}},
                ],
            }
        ]
        mock_client_factory.return_value = mock_client

        response = fantasy_client.get("/api/fantasy/matchup", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        data = response.json()
        assert data["degraded"] is True
        assert "not found" in data["message"].lower()

    @patch("backend.routers.fantasy.get_yahoo_client")
    def test_matchup_live_data_is_not_degraded(self, mock_client_factory, fantasy_client):
        """A real matchup with stats must NOT be flagged degraded (§R2)."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        mock_client = MagicMock(spec=YahooFantasyClient)
        mock_client.get_my_team_key.return_value = "388.l.123456.t.1"
        mock_client.get_scoreboard.return_value = [
            {
                "week": 12,
                "teams": [
                    {"team": {"team_key": "388.l.123456.t.1", "name": "My Team",
                              "team_stats": {"stats": [{"stat": {"stat_id": "50", "value": "7"}}]}}},
                    {"team": {"team_key": "388.l.123456.t.2", "name": "Opponent",
                              "team_stats": {"stats": [{"stat": {"stat_id": "50", "value": "5"}}]}}},
                ],
            }
        ]
        mock_client_factory.return_value = mock_client

        response = fantasy_client.get("/api/fantasy/matchup", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        assert response.json()["degraded"] is False
