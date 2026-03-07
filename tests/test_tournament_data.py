"""Tests for tournament_data.py -- A-26 Task 2."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from backend.services.tournament_data import (
    TournamentDataClient,
    get_tournament_client,
    fetch_tournament_bracket,
)


class TestTournamentDataClient:
    """Test TournamentDataClient functionality."""

    def setup_method(self):
        """Fresh client for each test."""
        self.client = TournamentDataClient()
        self.client._bracket_cache = {}
        self.client._cache_timestamp = None

    @patch('backend.services.tournament_data.requests.get')
    @patch.dict('os.environ', {'BALLDONTLIE_API_KEY': 'test_key'})
    def test_fetch_bracket_success(self, mock_get):
        """Successful API call returns seed dict."""
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "home_team": {"name": "Duke", "seed": "1"},
                    "away_team": {"name": "Vermont", "seed": "16"},
                },
                {
                    "home_team": {"name": "Kentucky", "seed": "3"},
                    "away_team": {"name": "Oakland", "seed": "14"},
                }
            ]
        }
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        result = self.client.fetch_bracket_data(season_year=2026)

        assert result["Duke"] == 1
        assert result["Vermont"] == 16
        assert result["Kentucky"] == 3
        assert result["Oakland"] == 14
        assert len(result) == 4

    @patch.dict('os.environ', {}, clear=True)
    def test_no_api_key_returns_empty(self):
        """Missing API key returns empty dict gracefully."""
        result = self.client.fetch_bracket_data()
        assert result == {}

    @patch('backend.services.tournament_data.requests.get')
    @patch.dict('os.environ', {'BALLDONTLIE_API_KEY': 'test_key'})
    def test_api_404_returns_empty(self, mock_get):
        """404 error (bracket not ready) returns empty dict."""
        from requests import HTTPError
        mock_resp = Mock()
        mock_resp.raise_for_status.side_effect = HTTPError(response=Mock(status_code=404))
        mock_get.return_value = mock_resp

        result = self.client.fetch_bracket_data()
        assert result == {}

    def test_caching(self):
        """Bracket data is cached for 6 hours."""
        self.client._bracket_cache = {"Duke": 1}
        self.client._cache_timestamp = datetime.utcnow()

        # Should return cached data without API call
        result = self.client.fetch_bracket_data()
        assert result == {"Duke": 1}

    def test_cache_expired(self):
        """Cache older than 6 hours triggers refetch."""
        from datetime import timedelta
        self.client._bracket_cache = {"Old Data": 1}
        self.client._cache_timestamp = datetime.utcnow() - timedelta(hours=7)

        # No API key, so should return empty after trying to refresh
        result = self.client.fetch_bracket_data()
        assert result == {}

    def test_get_team_seed_exact_match(self):
        """Exact team name match returns seed."""
        bracket = {"Duke": 1, "Vermont": 16}
        result = self.client.get_team_seed("Duke", bracket)
        assert result == 1

    def test_get_team_seed_substring_match(self):
        """Substring match works for partial names."""
        bracket = {"Duke Blue Devils": 1}
        result = self.client.get_team_seed("Duke", bracket)
        assert result == 1

    def test_get_team_seed_not_found(self):
        """Unknown team returns None."""
        bracket = {"Duke": 1}
        result = self.client.get_team_seed("Unknown Team", bracket)
        assert result is None

    def test_get_game_seeds(self):
        """Get seeds for both teams."""
        bracket = {"Duke": 1, "Vermont": 16}
        home, away = self.client.get_game_seeds(
            "Duke", "Vermont", bracket
        )
        assert home == 1
        assert away == 16


class TestTournamentDataSingleton:
    """Test singleton pattern."""

    def test_singleton_returns_same_instance(self):
        """get_tournament_client() returns same instance."""
        client1 = get_tournament_client()
        client2 = get_tournament_client()
        assert client1 is client2
