"""
Tests for v2-aligned MCMC simulator.

Verifies:
- 18 categories using lowercase canonical codes
- Win threshold = 10 (majority of 18)
- Legacy key normalization
- _PLAYER_WEEKLY_STD covers all categories
"""

import pytest

from backend.fantasy_baseball.mcmc_simulator import (
    simulate_weekly_matchup,
    simulate_roster_move,
    _normalize_category_key,
    _get_cat_score,
    _PLAYER_WEEKLY_STD,
    _CANONICAL_TO_LOWER,
    _LOWER_TO_CANONICAL,
)
from backend.stat_contract import (
    SCORING_CATEGORY_CODES,
    BATTING_CODES,
    PITCHING_CODES,
    LOWER_IS_BETTER,
)


def test_player_weekly_std_has_all_18_categories():
    """_PLAYER_WEEKLY_STD dict covers all 18 scoring categories (lowercase)."""
    expected_lower = {code.lower() for code in SCORING_CATEGORY_CODES}
    assert set(_PLAYER_WEEKLY_STD.keys()) == expected_lower


def test_normalize_category_key():
    """_normalize_category_key converts various inputs to lowercase v2 codes."""
    # Already lowercase v2 codes
    assert _normalize_category_key("k_p") == "k_p"
    assert _normalize_category_key("era") == "era"
    assert _normalize_category_key("hr_b") == "hr_b"

    # Legacy keys
    assert _normalize_category_key("k_pit") == "k_p"
    assert _normalize_category_key("k9") == "k_9"
    assert _normalize_category_key("hr") == "hr_b"  # Legacy batting HR

    # Uppercase canonical codes
    assert _normalize_category_key("K_P") == "k_p"
    assert _normalize_category_key("ERA") == "era"


def test_get_cat_score():
    """_get_cat_score extracts values from cat_scores with legacy key support."""
    cat_scores = {
        "k_p": 1.5,
        "era": -0.8,
        "hr_b": 2.0,
    }

    assert _get_cat_score(cat_scores, "k_p") == 1.5
    assert _get_cat_score(cat_scores, "era") == -0.8

    # Legacy key lookup
    cat_scores_legacy = {"k_pit": 1.5}
    assert _get_cat_score(cat_scores_legacy, "k_p") == 1.5

    # Missing key returns 0.0
    assert _get_cat_score(cat_scores, "w") == 0.0


def test_simulate_weekly_matchup_basic():
    """simulate_weekly_matchup works with v2 lowercase codes."""
    my_roster = [
        {
            "name": "Ohtani",
            "positions": ["DH", "SP"],
            "starts_this_week": 1,
            "cat_scores": {"r": 1.5, "hr_b": 2.0, "rbi": 1.8, "k_p": 1.2},
        },
        {
            "name": "Judge",
            "positions": ["OF"],
            "starts_this_week": 1,
            "cat_scores": {"r": 1.3, "hr_b": 2.5, "rbi": 1.6},
        },
    ]
    opp_roster = [
        {
            "name": "Trout",
            "positions": ["OF"],
            "starts_this_week": 1,
            "cat_scores": {"r": 1.0, "hr_b": 1.2, "rbi": 1.1},
        }
    ]

    result = simulate_weekly_matchup(my_roster, opp_roster, n_sims=1000)

    assert "win_prob" in result
    assert 0.0 <= result["win_prob"] <= 1.0
    assert result["n_sims"] == 1000
    assert len(result["category_win_probs"]) > 0
    assert result["expected_cats_won"] >= 0


def test_simulate_weekly_matchup_all_18_categories():
    """simulate_weekly_matchup handles all 18 v2 categories."""
    # Build roster with all 18 categories
    all_cats_lower = [c.lower() for c in SCORING_CATEGORY_CODES]

    my_roster = [
        {
            "name": "Super",
            "positions": ["DH"],
            "starts_this_week": 1,
            "cat_scores": {cat: 2.0 for cat in all_cats_lower},  # Dominant
        }
    ]
    opp_roster = [
        {
            "name": "Weak",
            "positions": ["DH"],
            "starts_this_week": 1,
            "cat_scores": {cat: -1.0 for cat in all_cats_lower},  # Below average
        }
    ]

    result = simulate_weekly_matchup(my_roster, opp_roster, n_sims=500)

    # Should win almost all simulations
    assert result["win_prob"] > 0.9
    assert len(result["category_win_probs"]) == 18
    assert result["expected_cats_won"] > 15


def test_simulate_weekly_matchup_empty_opponent():
    """Empty opponent roster uses league-average baseline."""
    my_roster = [
        {
            "name": "Player",
            "positions": ["1B"],
            "starts_this_week": 1,
            "cat_scores": {"r": 1.0, "hr_b": 1.0},
        }
    ]

    result = simulate_weekly_matchup(my_roster, [], n_sims=500)

    # Should be competitive against league average
    assert 0.2 <= result["win_prob"] <= 0.8
    assert result["categories_simulated"] == ["hr_b", "r"]  # Sorted alphabetically


def test_simulate_roster_move_v2():
    """simulate_roster_move works with v2 categories."""
    my_roster = [
        {
            "name": "Weak Player",
            "positions": ["OF"],
            "starts_this_week": 1,
            "cat_scores": {"r": -0.5, "hr_b": -0.3},
        }
    ]
    opp_roster = [
        {
            "name": "Opponent",
            "positions": ["OF"],
            "starts_this_week": 1,
            "cat_scores": {"r": 0.0, "hr_b": 0.0},
        }
    ]
    add_player = {
        "name": "Star Player",
        "positions": ["OF"],
        "starts_this_week": 1,
        "cat_scores": {"r": 2.0, "hr_b": 2.5},
    }

    result = simulate_roster_move(
        my_roster, opp_roster, add_player, "Weak Player", n_sims=500
    )

    assert "win_prob_before" in result
    assert "win_prob_after" in result
    assert "win_prob_gain" in result
    assert result["mcmc_enabled"] is True

    # Adding star player should improve win probability
    assert result["win_prob_after"] > result["win_prob_before"]
    assert result["win_prob_gain"] > 0


def test_lower_is_better_categories():
    """LOWER_IS_BETTER categories (ERA, WHIP, K_B, L, HR_P) are handled correctly.

    Note: z-scores for LOWER_IS_BETTER categories should be inverted before
    being passed to cat_scores. So a player with excellent ERA (low actual value)
    should have a POSITIVE z-score (higher = better in the z-score world).
    """
    # Create roster where I have better (lower) ERA/WHIP
    # Excellent ERA = lower actual value = HIGHER z-score (already inverted)
    my_roster = [
        {
            "name": "Ace",
            "positions": ["SP"],
            "starts_this_week": 1,
            "cat_scores": {"era": 2.0, "whip": 1.5},  # Excellent ERA/WHIP (positive z-scores)
        }
    ]
    opp_roster = [
        {
            "name": "Bum",
            "positions": ["SP"],
            "starts_this_week": 1,
            "cat_scores": {"era": -1.5, "whip": -1.0},  # Poor ERA/WHIP (negative z-scores)
        }
    ]

    result = simulate_weekly_matchup(my_roster, opp_roster, n_sims=500, categories=["era", "whip"])

    # I should win both categories most of the time (my z-scores are higher)
    assert result["category_win_probs"]["era"] > 0.7
    assert result["category_win_probs"]["whip"] > 0.6
    assert result["win_prob"] > 0.5  # Should win the matchup


def test_v2_codes_not_v1():
    """v2 lowercase codes are used, not v1 legacy keys."""
    # Check that the internal mappings use v2 codes
    assert "k_p" in _CANONICAL_TO_LOWER.values()
    assert "k_9" in _CANONICAL_TO_LOWER.values()
    assert "hr_b" in _CANONICAL_TO_LOWER.values()

    # v1 keys should NOT be primary (though legacy lookup works)
    # This is verified by the fact that _PLAYER_WEEKLY_STD uses v2 codes
    assert "k_pit" not in _PLAYER_WEEKLY_STD
    assert "k9" not in _PLAYER_WEEKLY_STD


def test_all_batting_and_pitching_codes_present():
    """All 9 batting and 9 pitching codes are in the lowercase mapping."""
    batting_lower = {code.lower() for code in BATTING_CODES}
    pitching_lower = {code.lower() for code in PITCHING_CODES}

    assert batting_lower.issubset(set(_PLAYER_WEEKLY_STD.keys()))
    assert pitching_lower.issubset(set(_PLAYER_WEEKLY_STD.keys()))


def test_simulate_weekly_matchup_legacy_keys():
    """simulate_weekly_matchup accepts legacy keys from cat_scores."""
    my_roster = [
        {
            "name": "Pitcher",
            "positions": ["SP"],
            "starts_this_week": 1,
            # Use legacy "k_pit" key
            "cat_scores": {"k_pit": 2.0, "w": 1.5},
        }
    ]
    opp_roster = [
        {
            "name": "Hitter",
            "positions": ["OF"],
            "starts_this_week": 1,
            "cat_scores": {"k_p": 0.0, "w": 0.0},
        }
    ]

    result = simulate_weekly_matchup(my_roster, opp_roster, n_sims=500)

    # Should detect and simulate "k_p" category (normalized from "k_pit")
    assert "k_p" in result["categories_simulated"]
    assert result["category_win_probs"]["k_p"] > 0.5  # My pitcher has better K projection


def test_simulate_weekly_matchup_no_categories():
    """Empty rosters return neutral 50/50 result."""
    result = simulate_weekly_matchup([], [], n_sims=100)

    assert result["win_prob"] == 0.5
    assert result["categories_simulated"] == []
    assert result["n_sims"] == 0


def test_win_threshold_10():
    """Win threshold is 10 (majority of 18 categories)."""
    # Create a scenario where we win exactly 10 categories
    my_scores = {}
    opp_scores = {}
    for i, cat in enumerate([c.lower() for c in SCORING_CATEGORY_CODES]):
        if i < 10:
            my_scores[cat] = 2.0  # Win first 10
            opp_scores[cat] = -1.0
        else:
            my_scores[cat] = -1.0  # Lose remaining 8
            opp_scores[cat] = 2.0

    my_roster = [{"name": "Team", "positions": ["DH"], "starts_this_week": 1, "cat_scores": my_scores}]
    opp_roster = [{"name": "Opp", "positions": ["DH"], "starts_this_week": 1, "cat_scores": opp_scores}]

    result = simulate_weekly_matchup(my_roster, opp_roster, n_sims=1000)

    # Should win most simulations (10-8 categories = 10+ threshold met)
    assert result["win_prob"] > 0.7
