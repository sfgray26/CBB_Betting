"""
EMAC-082: Tests for is_undroppable parsing fix in yahoo_client.py

Root cause: bool('0') == True in Python, so Yahoo's string '0' was
being parsed as undroppable. Fix uses explicit membership check.

Run:
    pytest tests/test_yahoo_client_undroppable.py -v
"""

import pytest
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient


def _build_player_list(is_undroppable_val):
    """Build a minimal Yahoo player list with is_undroppable set."""
    return [
        {
            "player_key": "mlb.p.1",
            "player_id": "1",
            "full_name": "Test Player",
            "editorial_team_abbr": "NYY",
            "is_undroppable": is_undroppable_val,
            "eligible_positions": [],
        }
    ]


def test_string_zero_is_droppable():
    """Yahoo returns '0' (string) — player should be droppable."""
    result = YahooFantasyClient._parse_player(_build_player_list('0'))
    assert result["is_undroppable"] is False


def test_string_one_is_undroppable():
    """Yahoo returns '1' (string) — player should be undroppable."""
    result = YahooFantasyClient._parse_player(_build_player_list('1'))
    assert result["is_undroppable"] is True


def test_int_zero_is_droppable():
    """Yahoo returns 0 (int) — player should be droppable."""
    result = YahooFantasyClient._parse_player(_build_player_list(0))
    assert result["is_undroppable"] is False


def test_int_one_is_undroppable():
    """Yahoo returns 1 (int) — player should be undroppable."""
    result = YahooFantasyClient._parse_player(_build_player_list(1))
    assert result["is_undroppable"] is True
