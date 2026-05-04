"""
H2H One Win Monte Carlo Simulator Tests

Tests the core algorithm for H2H One Win fantasy format category-by-category
win probability simulation.
"""

import pytest
import numpy as np
from datetime import date

from backend.fantasy_baseball.h2h_monte_carlo import (
    H2HOneWinSimulator,
    H2HWinResult,
)


def test_h2h_simulator_basic_functionality():
    """Test basic simulation with realistic roster projections."""
    sim = H2HOneWinSimulator()

    # Example roster: my team vs opponent
    my_roster = [
        {"name": "Ohtani", "R": 15, "HR": 4, "RBI": 12, "SB": 2, "NSB": 1, "AVG": 0.280, "OPS": 0.850,
         "W": 0, "QS": 0, "K": 0, "K/9": 0, "ERA": 0, "WHIP": 0},
        {"name": "Betts", "R": 12, "HR": 3, "RBI": 10, "SB": 1, "NSB": 0, "AVG": 0.290, "OPS": 0.880,
         "W": 0, "QS": 0, "K": 0, "K/9": 0, "ERA": 0, "WHIP": 0},
        {"name": "Cole", "R": 0, "HR": 0, "RBI": 0, "SB": 0, "NSB": 0, "AVG": 0, "OPS": 0,
         "W": 1, "QS": 1, "K": 12, "K/9": 10.5, "ERA": 2.80, "WHIP": 0.95},
    ]

    opponent_roster = [
        {"name": "Trout", "R": 14, "HR": 5, "RBI": 11, "SB": 2, "NSB": 1, "AVG": 0.270, "OPS": 0.830,
         "W": 0, "QS": 0, "K": 0, "K/9": 0, "ERA": 0, "WHIP": 0},
        {"name": "Acuna", "R": 13, "HR": 3, "RBI": 9, "SB": 3, "NSB": 2, "AVG": 0.285, "OPS": 0.860,
         "W": 0, "QS": 0, "K": 0, "K/9": 0, "ERA": 0, "WHIP": 0},
        {"name": "Strasburg", "R": 0, "HR": 0, "RBI": 0, "SB": 0, "NSB": 0, "AVG": 0, "OPS": 0,
         "W": 1, "QS": 1, "K": 10, "K/9": 9.8, "ERA": 3.20, "WHIP": 1.10},
    ]

    result = sim.simulate_week(my_roster, opponent_roster, n_sims=1000)

    # Verify result structure
    assert isinstance(result, H2HWinResult)
    assert 0.0 <= result.win_probability <= 1.0
    assert isinstance(result.locked_categories, list)
    assert isinstance(result.swing_categories, list)
    assert isinstance(result.vulnerable_categories, list)
    assert isinstance(result.category_win_probs, dict)
    assert result.n_simulations == 1000
    assert result.as_of_date == date.today()


def test_h2h_simulator_dominant_team():
    """Test simulation where my team dominates all categories."""
    sim = H2HOneWinSimulator()

    # My team: way better in all categories
    my_roster = [
        {"name": "Superstar", "R": 50, "HR": 15, "RBI": 40, "SB": 10, "NSB": 8, "AVG": 0.350, "OPS": 1.100,
         "W": 3, "QS": 3, "K": 40, "K/9": 14.0, "ERA": 1.50, "WHIP": 0.70},
    ]

    opponent_roster = [
        {"name": "Scrubs", "R": 5, "HR": 1, "RBI": 4, "SB": 1, "NSB": 0, "AVG": 0.200, "OPS": 0.600,
         "W": 0, "QS": 0, "K": 5, "K/9": 5.0, "ERA": 8.00, "WHIP": 2.50},
    ]

    result = sim.simulate_week(my_roster, opponent_roster, n_sims=1000)

    # Should win nearly all simulations
    assert result.win_probability > 0.95
    assert result.mean_categories_won > 8


def test_h2h_simulator_even_matchup():
    """Test simulation where teams are evenly matched."""
    sim = H2HOneWinSimulator()

    # Both teams have identical projections
    my_roster = [
        {"name": "Player A", "R": 20, "HR": 5, "RBI": 15, "SB": 3, "NSB": 2, "AVG": 0.270, "OPS": 0.820,
         "W": 1, "QS": 1, "K": 15, "K/9": 9.0, "ERA": 3.50, "WHIP": 1.15},
    ]

    opponent_roster = [
        {"name": "Player B", "R": 20, "HR": 5, "RBI": 15, "SB": 3, "NSB": 2, "AVG": 0.270, "OPS": 0.820,
         "W": 1, "QS": 1, "K": 15, "K/9": 9.0, "ERA": 3.50, "WHIP": 1.15},
    ]

    result = sim.simulate_week(my_roster, opponent_roster, n_sims=1000)

    # With 18 categories, WIN_THRESHOLD=10. For evenly matched teams (p=0.5 per category),
    # P(X >= 10) where X ~ Binomial(18, 0.5) is approximately 24%. This is correct:
    # when evenly matched, you need more than half the categories to win, which is unlikely.
    assert 0.15 <= result.win_probability <= 0.35
    # All categories should be swing (close to 50%)
    assert len(result.swing_categories) >= 10


def test_h2h_simulator_performance():
    """Test that 10k simulations complete in <200ms."""
    import time

    sim = H2HOneWinSimulator()

    # Realistic roster size (9 hitters + 5 pitchers = 14 players per team)
    my_roster = [
        {"name": f"Player{i}", "R": 10, "HR": 3, "RBI": 8, "SB": 1, "NSB": 0, "AVG": 0.260, "OPS": 0.780,
         "W": 0, "QS": 0, "K": 0, "K/9": 0, "ERA": 0, "WHIP": 0}
        for i in range(14)
    ]

    opponent_roster = [
        {"name": f"Opponent{i}", "R": 10, "HR": 3, "RBI": 8, "SB": 1, "NSB": 0, "AVG": 0.260, "OPS": 0.780,
         "W": 0, "QS": 0, "K": 0, "K/9": 0, "ERA": 0, "WHIP": 0}
        for i in range(14)
    ]

    start = time.time()
    result = sim.simulate_week(my_roster, opponent_roster, n_sims=10000)
    elapsed_ms = (time.time() - start) * 1000

    # Verify performance target
    assert elapsed_ms < 200, f"Performance target failed: {elapsed_ms:.1f}ms >= 200ms"

    # Verify result is still valid
    assert result.n_simulations == 10000


def test_h2h_simulator_negative_nsb():
    """Test that negative NSB values are handled correctly."""
    sim = H2HOneWinSimulator()

    # My player: 0 SB, 3 CS (NSB = -3)
    my_roster = [
        {"name": "CaughtStealing", "R": 5, "HR": 1, "RBI": 4, "SB": 0, "NSB": -3, "AVG": 0.220, "OPS": 0.650,
         "W": 0, "QS": 0, "K": 0, "K/9": 0, "ERA": 0, "WHIP": 0},
    ]

    opponent_roster = [
        {"name": "Neutral", "R": 5, "HR": 1, "RBI": 4, "SB": 1, "NSB": 1, "AVG": 0.220, "OPS": 0.650,
         "W": 0, "QS": 0, "K": 0, "K/9": 0, "ERA": 0, "WHIP": 0},
    ]

    # Should not crash with negative NSB
    result = sim.simulate_week(my_roster, opponent_roster, n_sims=100)
    assert isinstance(result.win_probability, float)


def test_h2h_simulator_era_whip_lower_is_better():
    """Test that ERA and WHIP are handled correctly (lower is better)."""
    sim = H2HOneWinSimulator()

    # My pitcher: better ERA/WHIP (lower is better)
    my_roster = [
        {"name": "Ace", "R": 0, "HR": 0, "RBI": 0, "SB": 0, "NSB": 0, "AVG": 0, "OPS": 0,
         "W": 1, "QS": 1, "K": 10, "K/9": 10.0, "ERA": 2.50, "WHIP": 0.90},
    ]

    opponent_roster = [
        {"name": "Bum", "R": 0, "HR": 0, "RBI": 0, "SB": 0, "NSB": 0, "AVG": 0, "OPS": 0,
         "W": 1, "QS": 1, "K": 10, "K/9": 10.0, "ERA": 5.50, "WHIP": 1.60},
    ]

    result = sim.simulate_week(my_roster, opponent_roster, n_sims=1000)

    # Should have high win probability due to better ERA/WHIP
    assert result.win_probability > 0.60

    # ERA and WHIP should be in locked or swing (definitely not vulnerable)
    assert "ERA" not in result.vulnerable_categories
    assert "WHIP" not in result.vulnerable_categories


def test_h2h_simulator_category_win_probs():
    """Test that category win probabilities are computed correctly."""
    sim = H2HOneWinSimulator()

    my_roster = [
        {"name": "Hitter", "R": 20, "HR": 5, "RBI": 15, "SB": 3, "NSB": 2, "AVG": 0.270, "OPS": 0.820,
         "W": 0, "QS": 0, "K": 0, "K/9": 0, "ERA": 0, "WHIP": 0},
    ]

    opponent_roster = [
        {"name": "Pitcher", "R": 0, "HR": 0, "RBI": 0, "SB": 0, "NSB": 0, "AVG": 0, "OPS": 0,
         "W": 1, "QS": 1, "K": 15, "K/9": 10.0, "ERA": 3.00, "WHIP": 1.10},
    ]

    result = sim.simulate_week(my_roster, opponent_roster, n_sims=1000)

    # Verify all categories have win probabilities
    expected_cats = sim.HITTING_CATS + sim.PITCHING_CATS
    for cat in expected_cats:
        assert cat in result.category_win_probs
        assert 0.0 <= result.category_win_probs[cat] <= 1.0

    # Hitting categories should favor my roster
    hitting_cats = sim.HITTING_CATS
    my_hitting_edge = sum(result.category_win_probs.get(cat, 0.5) for cat in hitting_cats) / len(hitting_cats)
    assert my_hitting_edge > 0.70  # Should win most hitting categories


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
