"""Tests for OddsAPIClient.parse_odds_for_game — odds parsing and field extraction."""

import pytest
from unittest.mock import patch
from backend.services.odds import OddsAPIClient


def _client():
    return OddsAPIClient(api_key="test-key")


def _base_game(**overrides):
    """Minimal valid game dict from The Odds API."""
    game = {
        "id": "abc123",
        "commence_time": "2026-03-20T19:00:00Z",
        "home_team": "Duke",
        "away_team": "UNC",
        "neutral_site": False,
        "bookmakers": [],
    }
    game.update(overrides)
    return game


# ---------------------------------------------------------------------------
# neutral_site / is_neutral
# ---------------------------------------------------------------------------

class TestNeutralSite:
    """parse_odds_for_game correctly maps neutral_site -> is_neutral."""

    def test_neutral_site_true_maps_to_is_neutral_true(self):
        """API neutral_site=True (NCAA tournament) -> is_neutral=True."""
        game = _base_game(neutral_site=True)
        result = _client().parse_odds_for_game(game)
        assert result["is_neutral"] is True

    def test_neutral_site_false_maps_to_is_neutral_false(self):
        """Regular-season home game -> is_neutral=False."""
        game = _base_game(neutral_site=False)
        result = _client().parse_odds_for_game(game)
        assert result["is_neutral"] is False

    def test_neutral_site_absent_defaults_to_false(self):
        """When neutral_site field is missing entirely, is_neutral=False."""
        game = _base_game()
        del game["neutral_site"]
        result = _client().parse_odds_for_game(game)
        assert result["is_neutral"] is False

    def test_neutral_site_truthy_int_coerced_to_bool(self):
        """API occasionally returns 1 instead of True — should coerce correctly."""
        game = _base_game(neutral_site=1)
        result = _client().parse_odds_for_game(game)
        assert result["is_neutral"] is True

    def test_is_neutral_present_in_result_keys(self):
        """is_neutral must be a top-level key in the returned dict."""
        result = _client().parse_odds_for_game(_base_game())
        assert "is_neutral" in result
