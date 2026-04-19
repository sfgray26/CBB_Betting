"""
Tests for mcmc_simulator.py — MCMC weekly matchup simulator.

All tests are deterministic (seeded RNG) and require only numpy + the
player_board module. No Yahoo API calls are made.
"""

import pytest
import numpy as np

from backend.fantasy_baseball.mcmc_simulator import (
    simulate_weekly_matchup,
    simulate_roster_move,
    _MCMC_DISABLED,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batter(name, hr=0.5, r=0.6, rbi=0.6, nsb=0.3, avg=0.4, ops=0.4, pos="OF"):
    return {
        "name": name,
        "positions": [pos],
        "starts_this_week": 1,
        "cat_scores": {"hr": hr, "r": r, "rbi": rbi, "nsb": nsb, "avg": avg, "ops": ops},
    }


def _pitcher(name, k_pit=0.8, era=0.5, whip=0.5, w=0.4, nsv=0.0, pos="SP", starts=1):
    return {
        "name": name,
        "positions": [pos],
        "starts_this_week": starts,
        "cat_scores": {"k_pit": k_pit, "era": era, "whip": whip, "w": w, "nsv": nsv},
    }


def _strong_roster(n_batters=7, n_pitchers=5, z=1.2):
    """A roster with uniformly above-average players."""
    return (
        [_batter(f"B{i}", hr=z, r=z, rbi=z, nsb=z, avg=z, ops=z) for i in range(n_batters)]
        + [_pitcher(f"P{i}", k_pit=z, era=z, whip=z, w=z) for i in range(n_pitchers)]
    )


def _weak_roster(n_batters=7, n_pitchers=5, z=-1.2):
    return (
        [_batter(f"B{i}", hr=z, r=z, rbi=z, nsb=z, avg=z, ops=z) for i in range(n_batters)]
        + [_pitcher(f"P{i}", k_pit=z, era=z, whip=z, w=z) for i in range(n_pitchers)]
    )


def _average_roster(n=12):
    return [_batter(f"Avg{i}", hr=0, r=0, rbi=0, nsb=0, avg=0, ops=0) for i in range(n)]


# ---------------------------------------------------------------------------
# Test 1: Strong roster beats average opponent at > 65% rate
# ---------------------------------------------------------------------------

def test_strong_roster_wins_more_than_average():
    my = _strong_roster(z=1.2)
    opp = _average_roster()
    result = simulate_weekly_matchup(my, opp, n_sims=2000, seed=42)
    assert result["win_prob"] > 0.65, f"Expected >65%, got {result['win_prob']:.2%}"


# ---------------------------------------------------------------------------
# Test 2: Weak roster loses more often than average
# ---------------------------------------------------------------------------

def test_weak_roster_loses_more_than_average():
    my = _weak_roster(z=-1.2)
    opp = _average_roster()
    result = simulate_weekly_matchup(my, opp, n_sims=2000, seed=42)
    assert result["win_prob"] < 0.35, f"Expected <35%, got {result['win_prob']:.2%}"


# ---------------------------------------------------------------------------
# Test 3: Symmetric rosters → win_prob near 0.50
# ---------------------------------------------------------------------------

def test_symmetric_rosters_near_50pct():
    roster = _strong_roster(z=0.5)
    result = simulate_weekly_matchup(roster, roster, n_sims=2000, seed=42)
    assert 0.35 < result["win_prob"] < 0.65, (
        f"Expected near 50%, got {result['win_prob']:.2%}"
    )


# ---------------------------------------------------------------------------
# Test 4: category_win_probs sums to expected_cats_won
# ---------------------------------------------------------------------------

def test_category_win_probs_consistent():
    my = _strong_roster()
    opp = _average_roster()
    result = simulate_weekly_matchup(my, opp, n_sims=1000, seed=99)
    total_from_probs = sum(result["category_win_probs"].values())
    assert abs(total_from_probs - result["expected_cats_won"]) < 0.05, (
        f"Category win probs sum ({total_from_probs:.2f}) != "
        f"expected_cats_won ({result['expected_cats_won']:.2f})"
    )


# ---------------------------------------------------------------------------
# Test 5: empty rosters return 50/50
# ---------------------------------------------------------------------------

def test_empty_roster_returns_50_pct():
    result = simulate_weekly_matchup([], [], n_sims=500, seed=0)
    assert result["win_prob"] == 0.5
    assert result["n_sims"] == 0


# ---------------------------------------------------------------------------
# Test 6: two-start pitchers contribute more counting stats
# ---------------------------------------------------------------------------

def test_two_start_pitcher_contributes_more():
    # Two identical pitchers, one makes 2 starts vs 1 start
    one_start = [_pitcher("Ace", k_pit=0.8, starts=1)]
    two_start = [_pitcher("Ace", k_pit=0.8, starts=2)]
    # Average opponent pitcher (k_pit=0.0 is z-score average)
    opp = [_pitcher("Opponent", k_pit=0.0, starts=1)]

    r1 = simulate_weekly_matchup(one_start, opp, n_sims=1000, seed=7)
    r2 = simulate_weekly_matchup(two_start, opp, n_sims=1000, seed=7)

    # Two-start pitcher should win K category more often
    # Note: category_win_probs keys are normalized to lowercase v2 codes (k_p, not k_pit)
    k1 = r1["category_win_probs"].get("k_p", 0.5)
    k2 = r2["category_win_probs"].get("k_p", 0.5)
    assert k2 > k1, f"Two-start K win prob ({k2:.2f}) should exceed one-start ({k1:.2f})"


# ---------------------------------------------------------------------------
# Test 7: simulate_roster_move — adding strong player improves win prob
# ---------------------------------------------------------------------------

def test_roster_move_improves_win_prob():
    my_roster = _weak_roster(z=-0.5)
    opp = _average_roster()
    add_player = _batter("Willi Castro", hr=0.8, r=1.0, rbi=0.9, nsb=0.5, avg=0.7, ops=0.7, pos="2B")
    drop_name = my_roster[0]["name"]  # drop the weakest batter

    result = simulate_roster_move(
        my_roster=my_roster,
        opponent_roster=opp,
        add_player=add_player,
        drop_player_name=drop_name,
        n_sims=1000,
        seed=42,
    )

    assert result["mcmc_enabled"] is True
    assert result["win_prob_after"] >= result["win_prob_before"], (
        f"Adding a strong player should not decrease win prob: "
        f"{result['win_prob_before']:.2%} -> {result['win_prob_after']:.2%}"
    )


# ---------------------------------------------------------------------------
# Test 8: win_prob_gain is after - before
# ---------------------------------------------------------------------------

def test_win_prob_gain_is_delta():
    my_roster = _strong_roster()
    opp = _average_roster()
    add = _batter("NewGuy", hr=1.5, r=1.5, rbi=1.5, nsb=1.0, avg=1.0, ops=1.0)
    drop = my_roster[0]["name"]

    result = simulate_roster_move(my_roster, opp, add, drop, n_sims=500, seed=1)
    expected = round(result["win_prob_after"] - result["win_prob_before"], 4)
    assert result["win_prob_gain"] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 9: Westburg 2B replacement — Willi Castro shows positive win_prob_gain
# ---------------------------------------------------------------------------

def test_westburg_2b_replacement_willi_castro_positive_gain():
    """
    Simulates the Westburg injury scenario: roster has a dead 2B slot (z=-1.5)
    and Willi Castro (z~+0.8) is available on waivers.
    Verifies that the move produces positive win_prob_gain.
    """
    # Current roster: strong everywhere except 2B (empty/injured slot)
    my_roster = (
        [_batter(f"S{i}", hr=0.8, r=0.8, rbi=0.8) for i in range(6)]
        + [_batter("Dead2B", hr=-1.5, r=-1.5, rbi=-1.5, nsb=-1.0, avg=-1.0, ops=-1.0, pos="2B")]
        + [_pitcher(f"P{i}", k_pit=0.7, era=0.7, whip=0.7, w=0.5) for i in range(4)]
    )
    opponent = _average_roster()

    willi_castro = _batter(
        "Willi Castro",
        hr=0.6, r=0.8, rbi=0.7, nsb=0.5, avg=0.5, ops=0.5,
        pos="2B",
    )

    result = simulate_roster_move(
        my_roster=my_roster,
        opponent_roster=opponent,
        add_player=willi_castro,
        drop_player_name="Dead2B",
        n_sims=1500,
        seed=2026,
    )

    assert result["mcmc_enabled"] is True
    assert result["win_prob_gain"] > 0, (
        f"Adding Willi Castro should improve win prob; got gain={result['win_prob_gain']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 10: result shape is correct
# ---------------------------------------------------------------------------

def test_simulate_weekly_matchup_shape():
    my = _strong_roster()
    opp = _average_roster()
    result = simulate_weekly_matchup(my, opp, n_sims=100, seed=0)

    assert "win_prob" in result
    assert "category_win_probs" in result
    assert "expected_cats_won" in result
    assert "n_sims" in result
    assert "elapsed_ms" in result
    assert "categories_simulated" in result
    assert isinstance(result["category_win_probs"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["category_win_probs"].values())


# ---------------------------------------------------------------------------
# Test 11: simulate_roster_move shape is correct
# ---------------------------------------------------------------------------

def test_simulate_roster_move_shape():
    my = _strong_roster()
    opp = _average_roster()
    add = _batter("AddGuy")
    result = simulate_roster_move(my, opp, add, my[0]["name"], n_sims=100, seed=0)

    required = [
        "win_prob_before", "win_prob_after", "win_prob_gain",
        "category_win_probs_before", "category_win_probs_after",
        "mcmc_enabled", "n_sims", "elapsed_ms",
    ]
    for key in required:
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Test 12: elapsed_ms is non-negative and reasonable
# ---------------------------------------------------------------------------

def test_elapsed_ms_under_500ms():
    my = _strong_roster()
    opp = _average_roster()
    result = simulate_weekly_matchup(my, opp, n_sims=1000)
    assert 0 <= result["elapsed_ms"] < 500, (
        f"Simulation too slow: {result['elapsed_ms']:.1f}ms"
    )
