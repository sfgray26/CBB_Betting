"""
Tests for MLBPlayerStats Pydantic V2 contract.

All tests use static dict fixtures -- no real API calls are made.
Ground truth: reports/K_A_BDL_STATS_SPEC.md + S19 live probe findings.
"""

import pytest
from backend.data_contracts.mlb_player_stats import MLBPlayerStats


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TEAM_DICT = {
    "id": 1,
    "slug": "los-angeles-dodgers",
    "abbreviation": "LAD",
    "display_name": "Los Angeles Dodgers",
    "short_display_name": "Dodgers",
    "name": "Dodgers",
    "location": "Los Angeles",
    "league": "National",
    "division": "West",
}

PLAYER_DICT = {
    "id": 12345,
    "first_name": "Freddie",
    "last_name": "Freeman",
    "full_name": "Freddie Freeman",
    "debut_year": 2010,
    "jersey": "5",
    "college": None,
    "position": "1B",
    "active": True,
    "birth_place": "Villa Park, CA",
    "dob": "12/9/1989",
    "age": 36,
    "height": "6'5\"",
    "weight": "220 lbs",
    "draft": "2007 ATL",
    "bats_throws": "L/R",
    "team": TEAM_DICT,
}

PITCHER_PLAYER_DICT = {
    "id": 67890,
    "first_name": "Clayton",
    "last_name": "Kershaw",
    "full_name": "Clayton Kershaw",
    "debut_year": 2008,
    "jersey": "22",
    "college": None,
    "position": "SP",
    "active": True,
    "birth_place": "Dallas, TX",
    "dob": "19/3/1988",
    "age": 38,
    "height": "6'3\"",
    "weight": "225 lbs",
    "draft": "2006 LAD",
    "bats_throws": "L/L",
    "team": TEAM_DICT,
}


# ---------------------------------------------------------------------------
# Test 1: Batter row (pitching fields null)
# ---------------------------------------------------------------------------

def test_batter_row_validates():
    """A typical batter stat row must validate with pitching fields absent."""
    raw = {
        "id": 1001,
        "player": PLAYER_DICT,
        "team": TEAM_DICT,
        "game_id": 9001,
        "season": 2026,
        # Batting fields
        "ab": 4,
        "r": 1,
        "h": 2,
        "double": 1,
        "triple": 0,
        "hr": 0,
        "rbi": 2,
        "bb": 1,
        "so": 0,
        "sb": 0,
        "cs": 0,
        "avg": 0.287,
        "obp": 0.362,
        "slg": 0.489,
        "ops": 0.851,
        # Pitching fields absent
    }
    stat = MLBPlayerStats.model_validate(raw)
    assert stat.ab == 4
    assert stat.r == 1
    assert stat.h == 2
    assert stat.double == 1
    assert stat.hr == 0
    assert stat.avg == 0.287
    assert stat.obp == 0.362
    assert stat.slg == 0.489
    assert stat.ops == 0.851
    # Pitching fields must be None (not validated-in as zero)
    assert stat.ip is None
    assert stat.era is None
    assert stat.whip is None
    assert stat.k is None


# ---------------------------------------------------------------------------
# Test 2: Pitcher row (batting fields null)
# ---------------------------------------------------------------------------

def test_pitcher_row_validates():
    """A typical pitcher stat row must validate with batting fields absent."""
    raw = {
        "id": 2002,
        "player": PITCHER_PLAYER_DICT,
        "team": TEAM_DICT,
        "game_id": 9002,
        "season": 2026,
        # Pitching fields
        "ip": "6.2",
        "h_allowed": 5,
        "r_allowed": 2,
        "er": 2,
        "bb_allowed": 1,
        "k": 9,
        "whip": 0.90,
        "era": 2.70,
        # Batting fields absent
    }
    stat = MLBPlayerStats.model_validate(raw)
    assert stat.ip == "6.2"
    assert stat.h_allowed == 5
    assert stat.r_allowed == 2
    assert stat.er == 2
    assert stat.bb_allowed == 1
    assert stat.k == 9
    assert stat.whip == 0.90
    assert stat.era == 2.70
    # Batting fields must be None
    assert stat.ab is None
    assert stat.avg is None
    assert stat.ops is None


# ---------------------------------------------------------------------------
# Test 3: Two-way player row (both sets populated)
# ---------------------------------------------------------------------------

def test_two_way_player_row_validates():
    """
    A two-way player row (e.g. Shohei Ohtani pitching + hitting on the same day)
    must validate with both batting and pitching fields present.
    """
    ohtani_player = {
        "id": 11111,
        "first_name": "Shohei",
        "last_name": "Ohtani",
        "full_name": "Shohei Ohtani",
        "debut_year": 2018,
        "jersey": "17",
        "college": None,
        "position": "SP/DH",
        "active": True,
        "birth_place": "Oshu, Japan",
        "dob": "5/7/1994",
        "age": 31,
        "height": "6'4\"",
        "weight": "210 lbs",
        "draft": None,
        "bats_throws": "L/R",
        "team": TEAM_DICT,
    }
    raw = {
        "id": 3003,
        "player": ohtani_player,
        "team": TEAM_DICT,
        "game_id": 9003,
        "season": 2026,
        # Batting
        "ab": 3,
        "r": 1,
        "h": 1,
        "double": 0,
        "triple": 0,
        "hr": 1,
        "rbi": 2,
        "bb": 1,
        "so": 1,
        "sb": 0,
        "cs": 0,
        "avg": 0.310,
        "obp": 0.400,
        "slg": 0.720,
        "ops": 1.120,
        # Pitching
        "ip": "6.0",
        "h_allowed": 4,
        "r_allowed": 1,
        "er": 1,
        "bb_allowed": 2,
        "k": 10,
        "whip": 1.00,
        "era": 1.50,
    }
    stat = MLBPlayerStats.model_validate(raw)
    # Both batting and pitching fields present
    assert stat.hr == 1
    assert stat.ops == 1.120
    assert stat.ip == "6.0"
    assert stat.era == 1.50
    assert stat.k == 10


# ---------------------------------------------------------------------------
# Test 4: Rate stats are float (not str)
# ---------------------------------------------------------------------------

def test_rate_stats_are_float():
    """
    S19 probe confirmed rate stats arrive as floats. The contract must accept them
    and expose them as float, not coerce from string.
    """
    raw = {
        "id": 4004,
        "player": PLAYER_DICT,
        "team": TEAM_DICT,
        "game_id": 9004,
        "avg": 0.333,
        "obp": 0.420,
        "slg": 0.567,
        "ops": 0.987,
        "whip": 1.12,
        "era": 3.25,
    }
    stat = MLBPlayerStats.model_validate(raw)
    assert isinstance(stat.avg, float)
    assert isinstance(stat.obp, float)
    assert isinstance(stat.slg, float)
    assert isinstance(stat.ops, float)
    assert isinstance(stat.whip, float)
    assert isinstance(stat.era, float)
    assert stat.avg == pytest.approx(0.333)


# ---------------------------------------------------------------------------
# Test 5: bdl_player_id property returns player.id
# ---------------------------------------------------------------------------

def test_bdl_player_id_property():
    """bdl_player_id convenience property must return player.id."""
    raw = {
        "player": PLAYER_DICT,
        "team": TEAM_DICT,
    }
    stat = MLBPlayerStats.model_validate(raw)
    assert stat.bdl_player_id == PLAYER_DICT["id"]
    assert stat.bdl_player_id == 12345


# ---------------------------------------------------------------------------
# Test 6: All stat fields are Optional -- empty stat row must not raise
# ---------------------------------------------------------------------------

def test_empty_stat_row_is_valid():
    """A row with only player/team must validate -- all stats Optional."""
    raw = {
        "player": PLAYER_DICT,
        "team": TEAM_DICT,
    }
    stat = MLBPlayerStats.model_validate(raw)
    assert stat.ab is None
    assert stat.avg is None
    assert stat.ip is None
    assert stat.era is None
    assert stat.game_id is None


# ---------------------------------------------------------------------------
# Test 7: ip field accepts string (not float)
# ---------------------------------------------------------------------------

def test_ip_is_string():
    """innings_pitched must be stored as a string like '6.2', never float."""
    raw = {
        "player": PITCHER_PLAYER_DICT,
        "team": TEAM_DICT,
        "ip": "6.2",
    }
    stat = MLBPlayerStats.model_validate(raw)
    assert stat.ip == "6.2"
    assert isinstance(stat.ip, str)
