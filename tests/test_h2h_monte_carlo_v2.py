"""
Tests for v2-aligned H2H Monte Carlo simulator.

Verifies:
- 18 categories using canonical codes
- Win threshold = 10 (majority of 18)
- LOWER_IS_BETTER from stat_contract
- simulate_week_from_projections() entry point
"""

import pytest

from backend.fantasy_baseball.h2h_monte_carlo import (
    H2HOneWinSimulator,
    H2HWinResult,
)
from backend.stat_contract import (
    SCORING_CATEGORY_CODES,
    BATTING_CODES,
    PITCHING_CODES,
    LOWER_IS_BETTER,
)


def test_simulator_v2_category_lists():
    """H2HOneWinSimulator uses v2 canonical category codes."""
    sim = H2HOneWinSimulator()

    # Should have 9 batting + 9 pitching = 18 total
    assert len(sim.HITTING_CATS) == 9
    assert len(sim.PITCHING_CATS) == 9
    assert set(sim.HITTING_CATS + sim.PITCHING_CATS) == SCORING_CATEGORY_CODES

    # HITTING_CATS should exactly match BATTING_CODES
    assert set(sim.HITTING_CATS) == BATTING_CODES

    # PITCHING_CATS should exactly match PITCHING_CODES
    assert set(sim.PITCHING_CATS) == PITCHING_CODES


def test_simulator_win_threshold():
    """Win threshold is 10 (majority of 18 categories)."""
    sim = H2HOneWinSimulator()
    assert sim.WIN_THRESHOLD == 10


def test_simulator_stat_cv_has_all_18_categories():
    """STAT_CV dict covers all 18 scoring categories."""
    sim = H2HOneWinSimulator()
    assert set(sim.STAT_CV.keys()) == SCORING_CATEGORY_CODES


def test_simulate_week_from_projections_basic():
    """simulate_week_from_projections() works with all 18 categories."""
    sim = H2HOneWinSimulator()

    # Create projections for all 18 categories
    my_finals = {code: 10.0 for code in SCORING_CATEGORY_CODES}
    opp_finals = {code: 5.0 for code in SCORING_CATEGORY_CODES}

    result = sim.simulate_week_from_projections(my_finals, opp_finals, n_sims=1000)

    # Should win almost all categories (higher is better for most)
    assert result.win_probability > 0.5
    assert result.mean_categories_won > 9.0

    # Verify structure
    assert isinstance(result, H2HWinResult)
    assert len(result.category_win_probs) == 18
    assert result.my_input_projections == my_finals
    assert result.opp_input_projections == opp_finals


def test_simulate_week_from_projections_missing_category_raises():
    """simulate_week_from_projections() validates all 18 categories present."""
    sim = H2HOneWinSimulator()

    # Missing "R"
    my_finals = {code: 10.0 for code in SCORING_CATEGORY_CODES if code != "R"}
    opp_finals = {code: 5.0 for code in SCORING_CATEGORY_CODES}

    with pytest.raises(ValueError, match="my_finals missing categories"):
        sim.simulate_week_from_projections(my_finals, opp_finals, n_sims=100)


def test_lower_is_better_categories():
    """LOWER_IS_BETTER categories reverse comparison logic."""
    sim = H2HOneWinSimulator()

    # Create projections where I'm higher (worse) in lower-is-better categories
    my_finals = {code: 10.0 for code in SCORING_CATEGORY_CODES}
    opp_finals = {code: 5.0 for code in SCORING_CATEGORY_CODES}

    result = sim.simulate_week_from_projections(my_finals, opp_finals, n_sims=5000)

    # For lower-is-better categories, higher = worse, so I should lose
    for cat in LOWER_IS_BETTER:
        assert result.category_win_probs[cat] < 0.5, f"{cat} should have <50% win prob when my value is higher"


def test_ratio_stats_in_simulation():
    """Ratio stats (AVG, OPS, ERA, WHIP, K_9) are simulated correctly."""
    sim = H2HOneWinSimulator()

    # Set up a close matchup in ratio stats
    my_finals = {
        "AVG": 0.280, "OPS": 0.850, "ERA": 3.50, "WHIP": 1.20, "K_9": 10.0,
        **{code: 0.0 for code in SCORING_CATEGORY_CODES if code not in {"AVG", "OPS", "ERA", "WHIP", "K_9"}}
    }
    opp_finals = {
        "AVG": 0.275, "OPS": 0.845, "ERA": 3.60, "WHIP": 1.25, "K_9": 9.5,
        **{code: 0.0 for code in SCORING_CATEGORY_CODES if code not in {"AVG", "OPS", "ERA", "WHIP", "K_9"}}
    }

    result = sim.simulate_week_from_projections(my_finals, opp_finals, n_sims=5000)

    # I should win most ratio categories (better AVG, OPS, ERA, WHIP, K_9)
    # ERA/WHIP are lower-is-better, and 3.50 < 3.60, 1.20 < 1.25 → I win
    ratio_cats = ["AVG", "OPS", "ERA", "WHIP", "K_9"]
    ratio_wins = sum(result.category_win_probs.get(cat, 0) > 0.5 for cat in ratio_cats)
    assert ratio_wins >= 3, "Should win most ratio categories with better projections"


def test_greenfield_categories_zero():
    """Greenfield categories (W, L, HR_P, NSV) work with zero values."""
    sim = H2HOneWinSimulator()

    # All projections zero (greenfield scenario)
    my_finals = {code: 0.0 for code in SCORING_CATEGORY_CODES}
    opp_finals = {code: 0.0 for code in SCORING_CATEGORY_CODES}

    result = sim.simulate_week_from_projections(my_finals, opp_finals, n_sims=1000)

    # All ties → each category gives 0.5 win → 9 categories won on average
    # Win threshold is 10, so with all ties, can never reach 10 → win_prob = 0
    # But mean_categories_won should be exactly 9.0 (18 * 0.5)
    assert result.win_probability == 0.0  # Need 10+ to win, but all ties = 9
    assert result.mean_categories_won == pytest.approx(9.0, abs=0.2)


def test_simulate_week_from_rosters():
    """Original simulate_week() still works with player rosters."""
    sim = H2HOneWinSimulator()

    # Create rosters with v2 canonical codes
    my_roster = [
        {"name": "Ohtani", "R": 15, "H": 25, "HR_B": 5, "RBI": 12, "TB": 40},
        {"name": "Judge", "R": 12, "H": 20, "HR_B": 6, "RBI": 14, "TB": 42},
    ]
    opp_roster = [
        {"name": "Trout", "R": 10, "H": 18, "HR_B": 3, "RBI": 10, "TB": 32},
        {"name": "Betts", "R": 11, "H": 19, "HR_B": 4, "RBI": 11, "TB": 35},
    ]

    result = sim.simulate_week(my_roster, opp_roster, n_sims=1000)

    # My roster should have advantage
    assert result.win_probability > 0.4  # Should be competitive
    assert len(result.category_win_probs) == 18


def test_classification_thresholds():
    """Category classification uses correct thresholds."""
    sim = H2HOneWinSimulator()

    # Dominant projections → locked categories (except lower-is-better where we're worse)
    my_finals = {code: 100.0 for code in SCORING_CATEGORY_CODES}
    opp_finals = {code: 1.0 for code in SCORING_CATEGORY_CODES}

    result = sim.simulate_week_from_projections(my_finals, opp_finals, n_sims=5000)

    # Should have many locked categories (>85% win prob)
    # Note: For lower-is-better (ERA, WHIP, K_B, L, HR_P), 100 > 1 means we lose
    locked_non_lower = [c for c in result.locked_categories if c not in LOWER_IS_BETTER]
    assert len(locked_non_lower) >= 9  # All non-lower-is-better should be locked

    # Lower-is-better categories should be vulnerable (we have 100 vs opp 1)
    assert len(result.vulnerable_categories) == len(LOWER_IS_BETTER)


def test_v2_codes_not_v1():
    """v2 codes are used, not v1 legacy codes."""
    sim = H2HOneWinSimulator()

    # v1 codes should NOT be present
    v1_codes = ["SB", "K", "K/9"]  # These became NSV, K_P, K_9
    for code in v1_codes:
        assert code not in sim.HITTING_CATS
        assert code not in sim.PITCHING_CATS
        assert code not in sim.STAT_CV

    # v2 replacements should be present
    assert "NSV" in sim.PITCHING_CATS
    assert "K_P" in sim.PITCHING_CATS
    assert "K_9" in sim.PITCHING_CATS
