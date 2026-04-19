"""
Statistical validation tests for the Monte Carlo simulation engine.

Proves that _draw_games, _percentiles, and simulate_player produce
mathematically correct distributions with appropriate convergence properties.
"""

import random
import statistics
from datetime import date
from unittest.mock import MagicMock

import pytest

from backend.services.simulation_engine import (
    CV,
    _draw_games,
    _percentiles,
    simulate_player,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hitter_row(
    bdl_player_id=1,
    as_of_date=None,
    games_in_window=10,
    w_games=10,
    w_runs=None,
    w_home_runs=3.0,
    w_rbi=10.0,
    w_stolen_bases=2.0,
    w_ab=40.0,
    w_hits=12.0,
    w_tb=None,
    w_doubles=None,
    w_triples=None,
    w_walks=None,
    w_net_stolen_bases=None,
    w_strikeouts_bat=None,
    w_ip=None,
    w_strikeouts_pit=None,
    w_earned_runs=None,
    w_hits_allowed=None,
    w_walks_allowed=None,
    w_qs=None,
):
    row = MagicMock()
    row.bdl_player_id = bdl_player_id
    row.as_of_date = as_of_date or date(2026, 4, 12)
    row.games_in_window = games_in_window
    row.w_games = w_games
    row.w_runs = w_runs
    row.w_home_runs = w_home_runs
    row.w_rbi = w_rbi
    row.w_stolen_bases = w_stolen_bases
    row.w_ab = w_ab
    row.w_hits = w_hits
    row.w_tb = w_tb
    row.w_doubles = w_doubles
    row.w_triples = w_triples
    row.w_walks = w_walks
    row.w_net_stolen_bases = w_net_stolen_bases
    row.w_strikeouts_bat = w_strikeouts_bat
    row.w_ip = w_ip
    row.w_strikeouts_pit = w_strikeouts_pit
    row.w_earned_runs = w_earned_runs
    row.w_hits_allowed = w_hits_allowed
    row.w_walks_allowed = w_walks_allowed
    row.w_qs = w_qs
    return row


def _make_pitcher_row(
    bdl_player_id=2,
    as_of_date=None,
    games_in_window=5,
    w_games=5,
    w_ip=30.0,
    w_strikeouts_pit=25.0,
    w_earned_runs=10.0,
    w_hits_allowed=20.0,
    w_walks_allowed=8.0,
    w_qs=None,
    w_k_per_9=None,
    w_ab=None,
    w_runs=None,
    w_home_runs=None,
    w_rbi=None,
    w_stolen_bases=None,
    w_hits=None,
    w_tb=None,
    w_doubles=None,
    w_triples=None,
    w_walks=None,
    w_net_stolen_bases=None,
    w_strikeouts_bat=None,
):
    row = MagicMock()
    row.bdl_player_id = bdl_player_id
    row.as_of_date = as_of_date or date(2026, 4, 12)
    row.games_in_window = games_in_window
    row.w_games = w_games
    row.w_ab = w_ab
    row.w_runs = w_runs
    row.w_home_runs = w_home_runs
    row.w_rbi = w_rbi
    row.w_stolen_bases = w_stolen_bases
    row.w_hits = w_hits
    row.w_tb = w_tb
    row.w_doubles = w_doubles
    row.w_triples = w_triples
    row.w_walks = w_walks
    row.w_net_stolen_bases = w_net_stolen_bases
    row.w_strikeouts_bat = w_strikeouts_bat
    row.w_ip = w_ip
    row.w_strikeouts_pit = w_strikeouts_pit
    row.w_earned_runs = w_earned_runs
    row.w_hits_allowed = w_hits_allowed
    row.w_walks_allowed = w_walks_allowed
    row.w_qs = w_qs
    row.w_k_per_9 = w_k_per_9
    return row


# ---------------------------------------------------------------------------
# Test 1: Mean convergence
# ---------------------------------------------------------------------------

def test_mean_convergence():
    """_draw_games mean over 5000 trials converges within 5% of expected."""
    rng = random.Random(42)
    rate = 0.3
    n_games = 130
    expected_mean = rate * n_games  # 39.0

    totals = [_draw_games(rng, rate, n_games) for _ in range(5000)]
    observed_mean = statistics.mean(totals)

    assert abs(observed_mean - expected_mean) / expected_mean < 0.05, (
        f"Mean {observed_mean:.2f} not within 5% of expected {expected_mean:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 2: Variance scales with games
# ---------------------------------------------------------------------------

def test_variance_scales_with_games():
    """Variance of 130-game sims should be ~2x variance of 65-game sims."""
    rng_65 = random.Random(42)
    rng_130 = random.Random(42)
    rate = 0.3
    n_trials = 3000

    totals_65 = [_draw_games(rng_65, rate, 65) for _ in range(n_trials)]
    totals_130 = [_draw_games(rng_130, rate, 130) for _ in range(n_trials)]

    var_65 = statistics.variance(totals_65)
    var_130 = statistics.variance(totals_130)

    ratio = var_130 / var_65
    assert 1.5 <= ratio <= 2.8, (
        f"Variance ratio {ratio:.2f} outside acceptable range [1.5, 2.8]"
    )


# ---------------------------------------------------------------------------
# Test 3: Percentile bracket coverage
# ---------------------------------------------------------------------------

def test_percentile_bracket_coverage():
    """~80% of season totals should fall between P10 and P90."""
    rng = random.Random(42)
    rate = 0.3
    n_games = 130
    n_trials = 5000

    totals = [_draw_games(rng, rate, n_games) for _ in range(n_trials)]
    p10, _p25, _p50, _p75, p90 = _percentiles(totals)

    in_bracket = sum(1 for t in totals if p10 <= t <= p90)
    fraction = in_bracket / n_trials

    assert 0.75 <= fraction <= 0.85, (
        f"Bracket coverage {fraction:.3f} outside acceptable range [0.75, 0.85]"
    )


# ---------------------------------------------------------------------------
# Test 4: Output CV matches input
# ---------------------------------------------------------------------------

def test_output_cv_matches_input():
    """Single-game draws at rate=1.0 should have sd/mean close to CV=0.35."""
    rng = random.Random(42)
    rate = 1.0
    n_trials = 10000

    draws = [_draw_games(rng, rate, 1) for _ in range(n_trials)]
    observed_mean = statistics.mean(draws)
    observed_sd = statistics.stdev(draws)
    observed_cv = observed_sd / observed_mean

    assert 0.25 <= observed_cv <= 0.40, (
        f"Observed CV {observed_cv:.3f} outside acceptable range [0.25, 0.40]"
    )


# ---------------------------------------------------------------------------
# Test 5: Zero rate produces zero projection
# ---------------------------------------------------------------------------

def test_zero_rate_produces_zero():
    """A hitter with w_home_runs=0 should project P50 and P90 HR at 0."""
    row = _make_hitter_row(w_home_runs=0.0)
    result = simulate_player(row, remaining_games=130, n_simulations=1000, seed=42)

    assert result.proj_hr_p50 == 0.0
    assert result.proj_hr_p90 == 0.0


# ---------------------------------------------------------------------------
# Test 6: High rate player has wider distribution
# ---------------------------------------------------------------------------

def test_high_rate_wider_distribution():
    """Power hitter (0.5 HR/game) should have wider P10-P90 spread than weak hitter (0.05 HR/game)."""
    power_row = _make_hitter_row(
        bdl_player_id=10,
        w_games=10,
        w_home_runs=5.0,   # 0.5 HR/game
        w_rbi=10.0,
        w_stolen_bases=1.0,
        w_ab=40.0,
        w_hits=12.0,
    )
    weak_row = _make_hitter_row(
        bdl_player_id=11,
        w_games=10,
        w_home_runs=0.5,   # 0.05 HR/game
        w_rbi=3.0,
        w_stolen_bases=1.0,
        w_ab=40.0,
        w_hits=10.0,
    )

    power_result = simulate_player(power_row, remaining_games=130, n_simulations=1000, seed=42)
    weak_result = simulate_player(weak_row, remaining_games=130, n_simulations=1000, seed=42)

    power_spread = power_result.proj_hr_p90 - power_result.proj_hr_p10
    weak_spread = weak_result.proj_hr_p90 - weak_result.proj_hr_p10

    assert power_spread > weak_spread, (
        f"Power spread {power_spread:.1f} should exceed weak spread {weak_spread:.1f}"
    )


# ---------------------------------------------------------------------------
# Test 7: Unknown player type returns empty result
# ---------------------------------------------------------------------------

def test_unknown_player_returns_empty():
    """A row with w_ab=None and w_ip=None should produce player_type='unknown' with None percentiles."""
    row = _make_hitter_row(w_ab=None, w_ip=None)
    # Also clear batting-dependent fields that MagicMock might auto-resolve
    row.w_hits = None
    row.w_home_runs = None
    row.w_rbi = None
    row.w_stolen_bases = None

    result = simulate_player(row, remaining_games=130, n_simulations=1000, seed=42)

    assert result.player_type == "unknown"
    assert result.proj_hr_p50 is None
    assert result.proj_hr_p90 is None
    assert result.proj_k_p50 is None
    assert result.proj_era_p50 is None


# ---------------------------------------------------------------------------
# Bonus: _percentiles edge cases
# ---------------------------------------------------------------------------

def test_percentiles_empty_list():
    """Empty list returns all zeros."""
    assert _percentiles([]) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_percentiles_single_value():
    """Single value list returns that value for all percentiles."""
    result = _percentiles([7.0])
    assert all(v == 7.0 for v in result)


def test_percentiles_ordering():
    """Percentiles should be monotonically non-decreasing."""
    rng = random.Random(42)
    values = [rng.gauss(50, 10) for _ in range(1000)]
    p10, p25, p50, p75, p90 = _percentiles(values)
    assert p10 <= p25 <= p50 <= p75 <= p90


# ---------------------------------------------------------------------------
# Bonus: Pitcher simulation produces sensible ERA/WHIP
# ---------------------------------------------------------------------------

def test_pitcher_simulation_produces_stats():
    """A pitcher row should produce non-None K, ERA, WHIP percentiles."""
    row = _make_pitcher_row()
    result = simulate_player(row, remaining_games=130, n_simulations=1000, seed=42)

    assert result.player_type == "pitcher"
    assert result.proj_k_p50 is not None
    assert result.proj_k_p50 > 0
    assert result.proj_era_p50 is not None
    assert result.proj_era_p50 > 0
    assert result.proj_whip_p50 is not None
    assert result.proj_whip_p50 > 0
    # Hitter fields should be None for pure pitcher
    assert result.proj_hr_p50 is None


def test_cv_constant():
    """CV constant should be 0.35."""
    assert CV == 0.35
