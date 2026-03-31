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


@patch("backend.fantasy_baseball.mcmc_calibration._get_player_board")
def test_convert_uses_board_when_available(mock_get_board):
    # Setup mock board
    mock_board = [
        {
            "name": "Mike Trout",
            "z_score": 2.5,
            "cat_scores": {"hr": 0.8, "r": 0.9, "rbi": 0.85},
        }
    ]
    mock_lookup = {"mike trout": mock_board[0]}
    mock_get_board.return_value = (mock_board, mock_lookup)
    
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
