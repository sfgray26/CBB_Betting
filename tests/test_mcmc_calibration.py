"""
Tests for mcmc_calibration.py — MCMC calibration layer.

These tests validate the conversion from Yahoo roster format
to MCMC simulator format, including proxy cat_score generation.
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.fantasy_baseball.mcmc_calibration import (
    _is_pitcher,
    _build_proxy_cat_scores,
    convert_yahoo_roster_to_mcmc_format,
    calculate_matchup_win_probability,
    BATTER_CATS,
    PITCHER_CATS,
)


def test_is_pitcher_detects_sp():
    assert _is_pitcher(["SP"]) is True
    assert _is_pitcher(["RP"]) is True
    assert _is_pitcher(["P"]) is True
    assert _is_pitcher(["OF"]) is False
    assert _is_pitcher(["1B", "OF"]) is False
    assert _is_pitcher(["SP", "RP"]) is True


def test_build_proxy_cat_scores_for_batter():
    scores = _build_proxy_cat_scores("Test Batter", ["OF"], total_z=1.0)
    
    # Should have all batter categories
    assert all(cat in scores for cat in BATTER_CATS)
    assert not any(cat in scores for cat in PITCHER_CATS)
    
    # All scores should be non-zero for positive total_z
    assert all(v != 0 for v in scores.values())
    
    # Scale should be applied (proxy scores are scaled down)
    assert all(abs(v) < 1.0 for v in scores.values())


def test_build_proxy_cat_scores_for_pitcher():
    scores = _build_proxy_cat_scores("Test Pitcher", ["SP"], total_z=1.5)
    
    # Should have all pitcher categories
    assert all(cat in scores for cat in PITCHER_CATS)
    assert not any(cat in scores for cat in BATTER_CATS)


def test_build_proxy_cat_scores_negative_z():
    scores = _build_proxy_cat_scores("Weak Player", ["OF"], total_z=-0.8)
    
    # All scores should be negative for negative total_z
    assert all(v <= 0 for v in scores.values())


def test_convert_yahoo_roster_basic():
    yahoo_roster = [
        {
            "name": "Mike Trout",
            "positions": ["OF", "CF"],
            "has_start": True,
        },
        {
            "name": "Shohei Ohtani",
            "positions": ["DH", "SP"],
            "has_start": True,
            "pitcher_slot": "SP",
        },
    ]
    
    result = convert_yahoo_roster_to_mcmc_format(yahoo_roster)
    
    assert len(result) == 2
    
    # Check batter
    trout = next(p for p in result if p["name"] == "Mike Trout")
    assert trout["positions"] == ["OF", "CF"]
    assert "cat_scores" in trout
    assert trout["starts_this_week"] == 1  # Non-pitcher
    
    # Check pitcher
    ohtani = next(p for p in result if p["name"] == "Shohei Ohtani")
    assert "SP" in ohtani["positions"]
    assert "cat_scores" in ohtani


def test_convert_yahoo_roster_empty():
    result = convert_yahoo_roster_to_mcmc_format([])
    assert result == []


def test_convert_yahoo_roster_skips_empty_names():
    yahoo_roster = [
        {"name": "", "positions": ["OF"]},
        {"name": "Valid Player", "positions": ["1B"]},
    ]
    
    result = convert_yahoo_roster_to_mcmc_format(yahoo_roster)
    assert len(result) == 1
    assert result[0]["name"] == "Valid Player"


@patch("backend.fantasy_baseball.mcmc_calibration._get_player_board_cached")
def test_convert_uses_board_when_available(mock_get_board_cached):
    # Setup mock board
    mock_board = [
        {
            "name": "Mike Trout",
            "z_score": 2.5,
            "cat_scores": {"hr": 0.8, "r": 0.9, "rbi": 0.85},
        }
    ]
    mock_lookup = {"mike trout": mock_board[0]}
    mock_get_board_cached.return_value = (mock_board, mock_lookup)
    
    yahoo_roster = [{"name": "Mike Trout", "positions": ["OF"]}]
    
    result = convert_yahoo_roster_to_mcmc_format(yahoo_roster)
    
    assert len(result) == 1
    assert result[0]["cat_scores"] == {"hr": 0.8, "r": 0.9, "rbi": 0.85}


@patch("backend.fantasy_baseball.mcmc_calibration._get_player_z_score_from_db")
def test_convert_uses_db_when_board_missing(mock_get_z):
    mock_get_z.return_value = 1.2
    
    yahoo_roster = [{"name": "Unknown Player", "positions": ["2B"]}]
    
    result = convert_yahoo_roster_to_mcmc_format(yahoo_roster)
    
    assert len(result) == 1
    assert "cat_scores" in result[0]
    mock_get_z.assert_called_once()


def test_calculate_matchup_win_probability_basic():
    my_roster = [
        {"name": "Player A", "positions": ["OF"]},
        {"name": "Player B", "positions": ["1B"]},
    ]
    opp_roster = [
        {"name": "Player C", "positions": ["OF"]},
        {"name": "Player D", "positions": ["2B"]},
    ]
    
    result = calculate_matchup_win_probability(
        my_roster, opp_roster, n_sims=100, seed=42
    )
    
    # Check result structure
    assert "win_prob" in result
    assert "category_win_probs" in result
    assert "expected_cats_won" in result
    assert "n_sims" in result
    
    # Win prob should be between 0 and 1
    assert 0 <= result["win_prob"] <= 1
    
    # Should have run the requested number of sims
    assert result["n_sims"] == 100


def test_calculate_matchup_returns_50_50_for_empty_rosters():
    result = calculate_matchup_win_probability([], [], n_sims=100, seed=42)
    assert result["win_prob"] == 0.5


@patch("backend.fantasy_baseball.mcmc_calibration.get_or_create_projection")
@patch("backend.fantasy_baseball.mcmc_calibration._get_player_z_score_from_db")
@patch("backend.fantasy_baseball.mcmc_calibration._get_player_board")
def test_tier3_fallback_produces_nonzero_cat_scores(mock_board, mock_db_z, mock_proj):
    """Unknown Tier-3 players must not get all-zero cat_scores."""
    mock_board.return_value = ([], {})          # Empty board — no Tier 1 match
    mock_db_z.return_value = None               # No DB data — no Tier 2 match
    mock_proj.return_value = {"z_score": 0.0, "cat_scores": {}}  # Tier 3 returns z=0

    batter = [{"name": "Ghost Batter", "positions": ["OF"]}]
    pitcher = [{"name": "Ghost Pitcher", "positions": ["SP"]}]

    batter_result = convert_yahoo_roster_to_mcmc_format(batter)
    pitcher_result = convert_yahoo_roster_to_mcmc_format(pitcher)

    assert len(batter_result) == 1
    assert len(pitcher_result) == 1

    # After fix: fallback z_score > 0 distributes non-zero values across all categories
    batter_scores = batter_result[0]["cat_scores"]
    pitcher_scores = pitcher_result[0]["cat_scores"]

    assert any(v != 0.0 for v in batter_scores.values()), (
        "Unknown batter should use _FALLBACK_BATTER_Z, not zero"
    )
    assert any(v != 0.0 for v in pitcher_scores.values()), (
        "Unknown pitcher should use _FALLBACK_PITCHER_Z, not zero"
    )

    # Batter should only have batter categories; pitcher only pitcher categories
    assert all(cat in BATTER_CATS for cat in batter_scores)
    assert all(cat in PITCHER_CATS for cat in pitcher_scores)


@patch("backend.fantasy_baseball.mcmc_calibration.get_or_create_projection")
@patch("backend.fantasy_baseball.mcmc_calibration._get_player_z_score_from_db")
@patch("backend.fantasy_baseball.mcmc_calibration._get_player_board")
def test_win_prob_not_frozen_at_0763(mock_board, mock_db_z, mock_proj):
    """Regression: constant win_prob=0.763 caused by all-zero opponent cat_scores.

    Scenario: my player has a strong board entry; all opponent players are
    unknown (Tier 3, z=0 before fix). After fix, opponent gets fallback z_score
    so win_prob must move away from the frozen 0.763 value.
    """
    strong_player = {
        "name": "Star Batter",
        "z_score": 18.0,
        "cat_scores": {
            "hr": 0.9, "r": 1.0, "rbi": 0.95,
            "nsb": 0.4, "avg": 0.5, "ops": 0.7,
            "tb": 0.6, "h": 0.4,
        },
    }
    mock_board.return_value = (
        [strong_player],
        {"star batter": strong_player},
    )
    mock_db_z.return_value = None
    mock_proj.return_value = {"z_score": 0.0, "cat_scores": {}}

    my_roster = [{"name": "Star Batter", "positions": ["OF"]}]
    opp_roster = [
        {"name": "Unknown Opp 1", "positions": ["OF"]},
        {"name": "Unknown Opp 2", "positions": ["1B"]},
        {"name": "Unknown Opp 3", "positions": ["2B"]},
    ]

    result = calculate_matchup_win_probability(
        my_roster=my_roster,
        opponent_roster=opp_roster,
        n_sims=1000,
        seed=42,
    )

    assert result["win_prob"] != pytest.approx(0.763, abs=0.001), (
        "win_prob frozen at 0.763 — opponent still has all-zero cat_scores"
    )
    # With 1 strong player (z=18) vs 3 fallback-z opponents (8 each, total z=24),
    # my team should LOSE more often (opponent has more total signal)
    assert result["win_prob"] < 0.5
    assert result["win_prob"] > 0.05  # But not absurdly weak
