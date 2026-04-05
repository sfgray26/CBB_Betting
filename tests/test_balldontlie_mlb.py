"""
Tests for BallDontLieClient MLB methods — Priority 3 validation.

Tests use captured fixtures (tests/fixtures/bdl_mlb_*.json) and
monkeypatching of _mlb_get to avoid live API calls.

Each test proves:
    - Return type is always a Pydantic-validated list (never raw dicts)
    - Pagination cursor is followed correctly
    - HTTP errors return empty list, never raise
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from backend.services.balldontlie import BallDontLieClient
from backend.data_contracts import MLBGame

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client():
    """BallDontLieClient with a dummy key — no live calls in these tests."""
    return BallDontLieClient(api_key="test-key-00000000-0000-0000-0000-000000000000")


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# get_mlb_games
# ---------------------------------------------------------------------------

class TestGetMlbGames:
    def test_returns_list_of_mlb_game(self, client):
        """Fixture-based: parses 19-game payload, returns list[MLBGame]."""
        fixture = load("bdl_mlb_games.json")
        with patch.object(client, "_mlb_get", return_value=fixture):
            result = client.get_mlb_games("2026-04-05")
        assert len(result) == 19
        assert all(isinstance(g, MLBGame) for g in result)

    def test_game_fields_are_validated(self, client):
        """Spot-check a known game from the fixture."""
        fixture = load("bdl_mlb_games.json")
        with patch.object(client, "_mlb_get", return_value=fixture):
            result = client.get_mlb_games("2026-04-05")
        game = result[0]
        assert game.id == 5057892
        assert game.season_type == "regular"
        assert game.home_team.abbreviation == "COL"
        assert game.away_team.abbreviation == "PHI"
        assert isinstance(game.home_team_data.inning_scores, list)

    def test_pagination_follows_cursor(self, client):
        """Two-page response: combines both pages into single list."""
        page1 = {
            "data": load("bdl_mlb_games.json")["data"][:5],
            "meta": {"per_page": 5, "next_cursor": 999},
        }
        page2 = {
            "data": load("bdl_mlb_games.json")["data"][5:8],
            "meta": {"per_page": 5},
        }
        responses = [page1, page2]
        call_count = 0

        def fake_mlb_get(path, params=None):
            nonlocal call_count
            r = responses[call_count]
            call_count += 1
            return r

        with patch.object(client, "_mlb_get", side_effect=fake_mlb_get):
            result = client.get_mlb_games("2026-04-05")

        assert call_count == 2
        assert len(result) == 8
        assert all(isinstance(g, MLBGame) for g in result)

    def test_http_error_returns_empty_list(self, client):
        """Any HTTP error → empty list, never raises."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "429 Too Many Requests", response=mock_response
        )

        def fake_mlb_get(path, params=None):
            raise requests.HTTPError("429 Too Many Requests", response=mock_response)

        with patch.object(client, "_mlb_get", side_effect=fake_mlb_get):
            result = client.get_mlb_games("2026-04-05")

        assert result == []

    def test_validation_error_returns_empty_list(self, client):
        """Corrupt payload (missing required field) → empty list, never raises."""
        corrupt = {"data": [{"id": "not-an-int"}], "meta": {"per_page": 25}}
        with patch.object(client, "_mlb_get", return_value=corrupt):
            result = client.get_mlb_games("2026-04-05")
        assert result == []

    def test_empty_date_returns_empty_list(self, client):
        """Date with no games → empty data array → empty list."""
        empty = {"data": [], "meta": {"per_page": 25}}
        with patch.object(client, "_mlb_get", return_value=empty):
            result = client.get_mlb_games("2020-01-01")
        assert result == []
