"""
Tests for game context extraction in player_mapper and PlayerCardResponse schema.

Tests cover:
- _build_player_game_context() extracts opponent_team, is_home, game_time, weather
- map_yahoo_player_to_canonical_row() propagates game context
- PlayerCardResponse schema exposes consumer-friendly field names
"""
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.services.player_mapper import _build_player_game_context, map_yahoo_player_to_canonical_row
from backend.contracts import PlayerGameContext
from backend.schemas import PlayerCardResponse


# ---------------------------------------------------------------------------
# _build_player_game_context
# ---------------------------------------------------------------------------

class TestBuildPlayerGameContext:
    def test_returns_none_when_no_opponent(self):
        result = _build_player_game_context({})
        assert result is None

    def test_returns_none_when_opponent_team_is_empty_string(self):
        result = _build_player_game_context({"opponent_team": "", "is_home": True})
        assert result is None

    def test_returns_context_when_opponent_team_present(self):
        result = _build_player_game_context({"opponent_team": "NYY", "is_home": True})
        assert isinstance(result, PlayerGameContext)
        assert result.opponent == "NYY"

    def test_is_home_true_maps_to_home(self):
        result = _build_player_game_context({"opponent_team": "BOS", "is_home": True})
        assert result.home_away == "home"

    def test_is_home_false_maps_to_away(self):
        result = _build_player_game_context({"opponent_team": "BOS", "is_home": False})
        assert result.home_away == "away"

    def test_is_home_missing_defaults_to_away(self):
        result = _build_player_game_context({"opponent_team": "ATL"})
        assert result.home_away == "away"

    def test_game_time_is_passed_through(self):
        game_time = datetime(2026, 5, 16, 19, 10, tzinfo=ZoneInfo("America/New_York"))
        result = _build_player_game_context({
            "opponent_team": "LAD",
            "is_home": True,
            "game_time": game_time,
        })
        assert result.game_time == game_time

    def test_game_time_is_none_when_absent(self):
        result = _build_player_game_context({"opponent_team": "LAD", "is_home": True})
        assert result.game_time is None

    def test_weather_is_passed_through(self):
        weather = {"temp_f": 72.0, "wind_mph": 5.0, "wind_direction": "out", "precip_chance": 0.0}
        result = _build_player_game_context({
            "opponent_team": "ATL",
            "is_home": True,
            "weather": weather,
        })
        assert result.weather is not None
        assert result.weather["temp_f"] == 72.0
        assert result.weather["wind_direction"] == "out"

    def test_weather_is_none_when_absent(self):
        result = _build_player_game_context({"opponent_team": "ATL", "is_home": True})
        assert result.weather is None


# ---------------------------------------------------------------------------
# map_yahoo_player_to_canonical_row — game context propagation
# ---------------------------------------------------------------------------

class TestMapYahooPlayerGameContextPropagation:
    def test_game_context_is_none_without_opponent_team(self, sample_yahoo_player):
        row = map_yahoo_player_to_canonical_row(sample_yahoo_player)
        assert row.game_context is None

    def test_game_context_propagates_when_opponent_team_present(self, sample_yahoo_player):
        sample_yahoo_player["opponent_team"] = "NYM"
        sample_yahoo_player["is_home"] = False
        row = map_yahoo_player_to_canonical_row(sample_yahoo_player)
        assert row.game_context is not None
        assert row.game_context.opponent == "NYM"
        assert row.game_context.home_away == "away"

    def test_game_context_includes_game_time_when_present(self, sample_yahoo_player):
        game_time = datetime(2026, 5, 17, 13, 5, tzinfo=ZoneInfo("America/New_York"))
        sample_yahoo_player["opponent_team"] = "CHC"
        sample_yahoo_player["is_home"] = True
        sample_yahoo_player["game_time"] = game_time
        row = map_yahoo_player_to_canonical_row(sample_yahoo_player)
        assert row.game_context.game_time == game_time

    def test_game_context_includes_weather_when_present(self, sample_yahoo_player):
        sample_yahoo_player["opponent_team"] = "SF"
        sample_yahoo_player["is_home"] = True
        sample_yahoo_player["weather"] = {"temp_f": 65.0, "wind_mph": 12.0, "wind_direction": "in", "precip_chance": 10.0}
        row = map_yahoo_player_to_canonical_row(sample_yahoo_player)
        assert row.game_context.weather is not None
        assert row.game_context.weather["wind_direction"] == "in"


# ---------------------------------------------------------------------------
# PlayerCardResponse schema
# ---------------------------------------------------------------------------

class TestPlayerCardResponseSchema:
    def test_schema_has_required_game_context_fields(self):
        card = PlayerCardResponse(
            player_key="mlb.p.12345",
            name="Test Player",
            team="LAD",
            opponent_team="NYY",
            game_time=None,
            is_home=True,
            weather=None,
        )
        assert card.player_key == "mlb.p.12345"
        assert card.opponent_team == "NYY"
        assert card.is_home is True
        assert card.weather is None

    def test_schema_allows_null_game_context_fields(self):
        card = PlayerCardResponse(
            player_key="mlb.p.99999",
            name="No Game Player",
            team="SEA",
            opponent_team=None,
            game_time=None,
            is_home=None,
            weather=None,
        )
        assert card.opponent_team is None
        assert card.is_home is None

    def test_schema_accepts_weather_dict(self):
        weather = {"temp_f": 78.0, "wind_mph": 3.0, "wind_direction": "out", "precip_chance": 0.0}
        card = PlayerCardResponse(
            player_key="mlb.p.11111",
            name="Batter Up",
            team="STL",
            opponent_team="CIN",
            game_time=datetime(2026, 5, 18, 18, 40, tzinfo=ZoneInfo("America/New_York")),
            is_home=False,
            weather=weather,
        )
        assert card.weather["temp_f"] == 78.0

    def test_schema_serializes_to_dict(self):
        card = PlayerCardResponse(
            player_key="mlb.p.54321",
            name="Ace",
            team="NYY",
            opponent_team="BOS",
            game_time=None,
            is_home=True,
            weather=None,
        )
        data = card.model_dump()
        assert "opponent_team" in data
        assert "is_home" in data
        assert "weather" in data
        assert "game_time" in data
