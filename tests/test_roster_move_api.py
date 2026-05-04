"""
Tests for Phase 4 Roster Move API endpoint.

Tests for POST /api/fantasy/roster/move.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def fantasy_client():
    with patch("backend.schedulers.fantasy_scheduler.start_fantasy_scheduler"):
        with patch("backend.schedulers.fantasy_scheduler.stop_fantasy_scheduler"):
            from backend.fantasy_app import app
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                yield client


class TestRosterMoveEndpoint:
    """Tests for POST /api/fantasy/roster/move endpoint."""

    def test_move_response_structure(self, fantasy_client):
        """Move response has all required fields."""
        # Mock the Yahoo client to return a test roster
        mock_roster = [
            {
                "player_key": "469.l.72586.p.12345",
                "name": "Test Player",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "BN",
            },
            {
                "player_key": "469.l.72586.p.67890",
                "name": "Another Player",
                "team": "BOS",
                "positions": ["OF"],
                "selected_position": "OF",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.12345"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "1B",
                },
            )

        assert response.status_code == 200
        data = response.json()

        # Response fields
        assert "success" in data
        assert "player_key" in data
        assert "from_position" in data
        assert "to_position" in data
        assert "message" in data
        assert "warnings" in data
        assert "freshness" in data

        # Freshness fields
        freshness = data["freshness"]
        assert "primary_source" in freshness
        assert "computed_at" in freshness
        assert "staleness_threshold_minutes" in freshness
        assert "is_stale" in freshness

    def test_successful_move(self, fantasy_client):
        """Player moved successfully."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.12345",
                "name": "Test Player",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "BN",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.12345"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "1B",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["player_key"] == "469.l.72586.p.12345"
        assert data["from_position"] == "BN"
        assert data["to_position"] == "1B"
        assert "Moved Test Player" in data["message"]

    def test_invalid_position(self, fantasy_client):
        """Invalid position returns error."""
        response = fantasy_client.post(
            "/api/fantasy/roster/move",
            json={
                "player_key": "469.l.72586.p.12345",
                "target_position": "INVALID",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Invalid position" in data["message"]

    def test_player_not_found(self, fantasy_client):
        """Player not on roster returns error."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.99999",
                "name": "Different Player",
                "team": "BOS",
                "positions": ["OF"],
                "selected_position": "OF",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "1B",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "not found on roster" in data["message"]

    def test_yahoo_api_error(self, fantasy_client):
        """Yahoo API error handled gracefully."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooAPIError

        mock_client = MagicMock()
        mock_client.get_roster.side_effect = YahooAPIError("API rate limit")

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "1B",
                },
            )

        assert response.status_code in (200, 502)  # Either graceful or HTTP exception

    def test_move_to_il(self, fantasy_client):
        """Player moved to IL slot."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.12345",
                "name": "Injured Player",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "1B",
                "status": "DTD",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.12345"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "IL",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["from_position"] == "1B"
        assert data["to_position"] == "IL"

    def test_valid_positions_accepted(self, fantasy_client):
        """All valid positions are accepted."""
        valid_positions = [
            "C", "1B", "2B", "3B", "SS", "OF", "Util",
            "SP", "RP", "P", "BN", "IL", "IL60",
        ]

        for pos in valid_positions:
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": pos,
                },
            )

            # Should not return "Invalid position" error
            # (may fail for other reasons like player not found)
            if response.status_code == 200:
                data = response.json()
                if not data.get("success") and "Invalid position" in data.get("message", ""):
                    assert False, f"Position {pos} was rejected as invalid"
