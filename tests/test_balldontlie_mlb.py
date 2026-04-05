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
from backend.data_contracts import MLBBettingOdd, MLBGame, MLBInjury, MLBPlayer

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


# ---------------------------------------------------------------------------
# get_mlb_odds
# ---------------------------------------------------------------------------

class TestGetMlbOdds:
    def test_returns_list_of_mlb_betting_odd(self, client):
        """Fixture-based: parses 6-vendor odds payload, returns list[MLBBettingOdd]."""
        fixture = load("bdl_mlb_odds.json")
        with patch.object(client, "_mlb_get", return_value=fixture):
            result = client.get_mlb_odds(5057892)
        assert len(result) == 6
        assert all(isinstance(o, MLBBettingOdd) for o in result)

    def test_spread_values_are_strings(self, client):
        """Contract enforcement: spread/total values arrive as strings."""
        fixture = load("bdl_mlb_odds.json")
        with patch.object(client, "_mlb_get", return_value=fixture):
            result = client.get_mlb_odds(5057892)
        for odd in result:
            assert isinstance(odd.spread_home_value, str)
            assert isinstance(odd.total_value, str)

    def test_float_properties_accessible(self, client):
        """Float properties work on the returned models."""
        fixture = load("bdl_mlb_odds.json")
        with patch.object(client, "_mlb_get", return_value=fixture):
            result = client.get_mlb_odds(5057892)
        fanduel = next(o for o in result if o.vendor == "fanduel")
        assert fanduel.spread_home_float == 1.5
        assert fanduel.total_float == 3.5

    def test_game_ids_param_sent_correctly(self, client):
        """Verifies game_ids[] param is passed (not game_id)."""
        fixture = load("bdl_mlb_odds.json")
        captured_params = {}

        def fake_get(path, params=None):
            captured_params.update(params or {})
            return fixture

        with patch.object(client, "_mlb_get", side_effect=fake_get):
            client.get_mlb_odds(5057892)

        assert "game_ids[]" in captured_params
        assert captured_params["game_ids[]"] == 5057892

    def test_http_error_returns_empty_list(self, client):
        """Any HTTP error → empty list, never raises."""
        def fake_get(path, params=None):
            raise requests.HTTPError("403 Forbidden")

        with patch.object(client, "_mlb_get", side_effect=fake_get):
            result = client.get_mlb_odds(5057892)

        assert result == []

    def test_no_odds_for_game_returns_empty_list(self, client):
        """Pre-game or no market → empty data → empty list."""
        empty = {"data": [], "meta": {"per_page": 25}}
        with patch.object(client, "_mlb_get", return_value=empty):
            result = client.get_mlb_odds(9999999)
        assert result == []


# ---------------------------------------------------------------------------
# get_mlb_injuries
# ---------------------------------------------------------------------------

class TestGetMlbInjuries:
    def test_returns_list_of_mlb_injury(self, client):
        """Fixture-based: parses first page (25 items) then stops on empty final page."""
        fixture = load("bdl_mlb_injuries.json")  # has next_cursor=409031
        final_page = {"data": [], "meta": {"per_page": 25}}  # no cursor = done

        responses = [fixture, final_page]
        call_count = 0

        def fake_get(path, params=None):
            nonlocal call_count
            r = responses[min(call_count, 1)]
            call_count += 1
            return r

        with patch.object(client, "_mlb_get", side_effect=fake_get):
            result = client.get_mlb_injuries()
        assert len(result) == 25
        assert all(isinstance(i, MLBInjury) for i in result)

    def test_injury_player_is_validated(self, client):
        """Nested MLBPlayer inside MLBInjury is fully validated."""
        fixture = load("bdl_mlb_injuries.json")
        with patch.object(client, "_mlb_get", return_value=fixture):
            result = client.get_mlb_injuries()
        first = result[0]
        assert first.player.full_name == "Travis Adams"
        assert first.status == "15-Day-IL"
        assert first.type == "Triceps"

    def test_nullable_fields_handled(self, client):
        """detail and side can be None — both appear in live fixture."""
        fixture = load("bdl_mlb_injuries.json")
        with patch.object(client, "_mlb_get", return_value=fixture):
            result = client.get_mlb_injuries()
        details = [i.detail for i in result]
        sides = [i.side for i in result]
        assert None in details
        assert None in sides

    def test_pagination_fetches_all_pages(self, client):
        """Cursor from page 1 triggers page 2 fetch — both pages combined."""
        fixture = load("bdl_mlb_injuries.json")  # has next_cursor=409031
        page2 = {
            "data": fixture["data"][:3],
            "meta": {"per_page": 25},           # no next_cursor = last page
        }
        responses = [fixture, page2]
        call_count = 0

        def fake_get(path, params=None):
            nonlocal call_count
            r = responses[call_count]
            call_count += 1
            return r

        with patch.object(client, "_mlb_get", side_effect=fake_get):
            result = client.get_mlb_injuries()

        assert call_count == 2
        assert len(result) == 28  # 25 + 3

    def test_http_error_returns_empty_list(self, client):
        """HTTP error on first page → empty list, never raises."""
        def fake_get(path, params=None):
            raise requests.HTTPError("500 Internal Server Error")

        with patch.object(client, "_mlb_get", side_effect=fake_get):
            result = client.get_mlb_injuries()

        assert result == []

    def test_http_error_on_page2_returns_partial_results(self, client):
        """Error mid-pagination returns what was collected so far (logged)."""
        fixture = load("bdl_mlb_injuries.json")  # has next_cursor
        call_count = 0

        def fake_get(path, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fixture
            raise requests.HTTPError("503 Service Unavailable")

        with patch.object(client, "_mlb_get", side_effect=fake_get):
            result = client.get_mlb_injuries()

        # First page (25 items) was collected before error on page 2
        assert len(result) == 25
        assert all(isinstance(i, MLBInjury) for i in result)


# ---------------------------------------------------------------------------
# search_mlb_players / get_mlb_player
# ---------------------------------------------------------------------------

class TestMlbPlayers:
    def test_search_returns_list_of_mlb_player(self, client):
        """Fixture-based: Ohtani search returns 1 player."""
        fixture = load("bdl_mlb_players.json")
        with patch.object(client, "_mlb_get", return_value=fixture):
            result = client.search_mlb_players("Ohtani")
        assert len(result) == 1
        assert all(isinstance(p, MLBPlayer) for p in result)
        assert result[0].full_name == "Shohei Ohtani"
        assert result[0].team.abbreviation == "LAD"

    def test_search_player_fields(self, client):
        """Nullable fields (college, draft) correctly null."""
        fixture = load("bdl_mlb_players.json")
        with patch.object(client, "_mlb_get", return_value=fixture):
            result = client.search_mlb_players("Ohtani")
        p = result[0]
        assert p.college is None
        assert p.draft is None
        assert p.dob == "5/7/1994"  # D/M/YYYY format preserved as-is

    def test_search_http_error_returns_empty(self, client):
        """HTTP error → empty list, never raises."""
        def fake_get(path, params=None):
            raise requests.HTTPError("404 Not Found")

        with patch.object(client, "_mlb_get", side_effect=fake_get):
            result = client.search_mlb_players("zzzzunknown")

        assert result == []

    def test_get_player_by_id_direct_object(self, client):
        """get_mlb_player handles direct object response (no data wrapper)."""
        fixture = load("bdl_mlb_players.json")
        raw_player = fixture["data"][0]  # plain player dict
        with patch.object(client, "_mlb_get", return_value=raw_player):
            result = client.get_mlb_player(208)
        assert isinstance(result, MLBPlayer)
        assert result.full_name == "Shohei Ohtani"

    def test_get_player_by_id_wrapped_response(self, client):
        """get_mlb_player handles data-wrapped response."""
        fixture = load("bdl_mlb_players.json")
        wrapped = {"data": fixture["data"][0]}
        with patch.object(client, "_mlb_get", return_value=wrapped):
            result = client.get_mlb_player(208)
        assert isinstance(result, MLBPlayer)

    def test_get_player_by_id_not_found(self, client):
        """HTTP error on player lookup → None, never raises."""
        def fake_get(path, params=None):
            raise requests.HTTPError("404 Not Found")

        with patch.object(client, "_mlb_get", side_effect=fake_get):
            result = client.get_mlb_player(9999999)

        assert result is None
