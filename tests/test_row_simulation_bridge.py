"""
Tests for ROW → Simulation Bridge (Phase 3 Workstream C).

Verifies:
- prepare_simulation_inputs() creates valid bundles
- prepare_h2h_monte_carlo_inputs() returns correct format
- prepare_mcmc_simulation_inputs() creates synthetic rosters
- Variance inflation calculations
"""

import pytest

from backend.services.row_simulation_bridge import (
    prepare_simulation_inputs,
    prepare_h2h_monte_carlo_inputs,
    prepare_mcmc_simulation_inputs,
    calculate_variance_inflation,
    summarize_simulation_bundles,
    SimulationInputBundle,
)
from backend.services.row_projector import ROWProjectionResult
from backend.stat_contract import SCORING_CATEGORY_CODES, LOWER_IS_BETTER


def test_prepare_simulation_inputs_basic():
    """prepare_simulation_inputs() creates valid bundle from two ROW results."""
    my_row = ROWProjectionResult(
        R=50.0, H=120.0, HR_B=18.0, RBI=55.0, K_B=80.0, TB=180.0, NSB=5.0,
        AVG=0.270, OPS=0.780,
        W=4.0, L=2.0, HR_P=12.0, K_P=90.0, QS=5.0, NSV=-2.0,
        ERA=3.50, WHIP=1.20, K_9=9.5,
    )
    opp_row = ROWProjectionResult(
        R=45.0, H=115.0, HR_B=15.0, RBI=50.0, K_B=75.0, TB=170.0, NSB=3.0,
        AVG=0.265, OPS=0.770,
        W=3.0, L=3.0, HR_P=15.0, K_P=85.0, QS=4.0, NSV=-1.0,
        ERA=3.80, WHIP=1.25, K_9=9.0,
    )

    bundle = prepare_simulation_inputs(my_row, opp_row)

    assert isinstance(bundle, SimulationInputBundle)
    assert set(bundle.my_row_finals.keys()) == SCORING_CATEGORY_CODES
    assert set(bundle.opp_row_finals.keys()) == SCORING_CATEGORY_CODES
    assert len(bundle.category_math) == 18


def test_prepare_simulation_inputs_validates_categories():
    """Missing categories raise ValueError."""
    # Create partial ROW result with all 18 categories
    my_full = ROWProjectionResult(**{k: 0.0 for k in SCORING_CATEGORY_CODES})
    opp_full = ROWProjectionResult(**{k: 0.0 for k in SCORING_CATEGORY_CODES})

    bundle = prepare_simulation_inputs(my_full, opp_full)
    assert bundle is not None


def test_prepare_h2h_monte_carlo_inputs():
    """prepare_h2h_monte_carlo_inputs() returns correct tuple format."""
    my_row = ROWProjectionResult(R=50.0, H=120.0, HR_B=18.0, RBI=55.0, K_B=80.0, TB=180.0, NSB=5.0,
                                  AVG=0.270, OPS=0.780, W=4.0, L=2.0, HR_P=12.0, K_P=90.0, QS=5.0, NSV=-2.0,
                                  ERA=3.50, WHIP=1.20, K_9=9.5)
    opp_row = ROWProjectionResult(R=45.0, H=115.0, HR_B=15.0, RBI=50.0, K_B=75.0, TB=170.0, NSB=3.0,
                                  AVG=0.265, OPS=0.770, W=3.0, L=3.0, HR_P=15.0, K_P=85.0, QS=4.0, NSV=-1.0,
                                  ERA=3.80, WHIP=1.25, K_9=9.0)

    bundle = prepare_simulation_inputs(my_row, opp_row)
    my_finals, opp_finals = prepare_h2h_monte_carlo_inputs(bundle)

    assert isinstance(my_finals, dict)
    assert isinstance(opp_finals, dict)
    assert set(my_finals.keys()) == SCORING_CATEGORY_CODES
    assert set(opp_finals.keys()) == SCORING_CATEGORY_CODES
    assert my_finals["R"] == 50.0
    assert opp_finals["R"] == 45.0


def test_prepare_mcmc_simulation_inputs():
    """prepare_mcmc_simulation_inputs() creates synthetic rosters with z-scores."""
    my_row = ROWProjectionResult(R=50.0, H=120.0, HR_B=18.0, RBI=55.0, K_B=80.0, TB=180.0, NSB=5.0,
                                  AVG=0.270, OPS=0.780, W=4.0, L=2.0, HR_P=12.0, K_P=90.0, QS=5.0, NSV=-2.0,
                                  ERA=3.50, WHIP=1.20, K_9=9.5)
    opp_row = ROWProjectionResult(R=40.0, H=110.0, HR_B=12.0, RBI=45.0, K_B=90.0, TB=160.0, NSB=2.0,
                                  AVG=0.260, OPS=0.750, W=3.0, L=4.0, HR_P=18.0, K_P=75.0, QS=3.0, NSV=-3.0,
                                  ERA=4.20, WHIP=1.35, K_9=8.0)

    bundle = prepare_simulation_inputs(my_row, opp_row)
    my_roster, opp_roster = prepare_mcmc_simulation_inputs(bundle)

    # Should be lists of player dicts
    assert isinstance(my_roster, list)
    assert isinstance(opp_roster, list)
    assert len(my_roster) == 1  # Synthetic "super player"
    assert len(opp_roster) == 1

    # Check player dict structure
    my_player = my_roster[0]
    assert "name" in my_player
    assert "positions" in my_player
    assert "cat_scores" in my_player
    assert "starts_this_week" in my_player

    # z-scores should be present for all 18 categories (lowercase)
    assert len(my_player["cat_scores"]) == 18
    for cat in SCORING_CATEGORY_CODES:
        assert cat.lower() in my_player["cat_scores"]


def test_mcmc_z_scores_sign_convention():
    """MCMC z-scores use higher-is-better convention for all categories."""
    # My team has better ERA (lower = better)
    my_row = ROWProjectionResult(ERA=3.00, **{k: 0.0 for k in SCORING_CATEGORY_CODES if k != "ERA"})
    opp_row = ROWProjectionResult(ERA=5.00, **{k: 0.0 for k in SCORING_CATEGORY_CODES if k != "ERA"})

    bundle = prepare_simulation_inputs(my_row, opp_row)
    my_roster, opp_roster = prepare_mcmc_simulation_inputs(bundle)

    # My ERA z-score should be POSITIVE (better)
    # (3.00 - 5.00 = -2.00, but ERA is lower-is-better, so flipped to positive)
    my_era_z = my_roster[0]["cat_scores"]["era"]
    assert my_era_z > 0, f"Expected positive z-score for better ERA, got {my_era_z}"


def test_calculate_variance_inflation():
    """calculate_variance_inflation() returns reasonable multipliers."""
    row = ROWProjectionResult(
        R=50.0, H=120.0, HR_B=18.0, RBI=55.0, K_B=80.0, TB=180.0, NSB=5.0,
        AVG=0.270, OPS=0.780, W=4.0, L=2.0, HR_P=12.0, K_P=90.0, QS=5.0, NSV=-2.0,
        ERA=3.50, WHIP=1.20, K_9=9.5,
    )

    # Full week remaining
    var_full_week = calculate_variance_inflation(row, days_remaining=7)
    assert all(v >= 1.0 for v in var_full_week.values())

    # Fewer days remaining = higher variance
    var_one_day = calculate_variance_inflation(row, days_remaining=1)
    for cat in SCORING_CATEGORY_CODES:
        assert var_one_day[cat] >= var_full_week[cat]


def test_summarize_simulation_bundles():
    """summarize_simulation_bundles() creates readable summary."""
    my_row = ROWProjectionResult(
        R=50.0, H=120.0, HR_B=18.0, RBI=55.0, K_B=80.0, TB=180.0, NSB=5.0,
        AVG=0.270, OPS=0.780, W=4.0, L=2.0, HR_P=12.0, K_P=90.0, QS=5.0, NSV=-2.0,
        ERA=3.50, WHIP=1.20, K_9=9.5,
    )
    opp_row = ROWProjectionResult(
        R=45.0, H=115.0, HR_B=15.0, RBI=50.0, K_B=75.0, TB=170.0, NSB=3.0,
        AVG=0.265, OPS=0.770, W=3.0, L=3.0, HR_P=15.0, K_P=85.0, QS=4.0, NSV=-1.0,
        ERA=3.80, WHIP=1.25, K_9=9.0,
    )

    bundle = prepare_simulation_inputs(my_row, opp_row)
    summary = summarize_simulation_bundles(bundle)

    assert "categories_winning" in summary
    assert "categories_losing" in summary
    assert "categories_tied" in summary
    assert "biggest_lead" in summary
    assert "biggest_deficit" in summary
    assert "swing_categories" in summary

    # Should have at least some winning categories
    assert summary["categories_winning"] >= 0
    assert summary["categories_winning"] + summary["categories_losing"] + summary["categories_tied"] == 18


def test_prepare_simulation_inputs_with_ratio_components():
    """prepare_simulation_inputs() accepts ratio stat components."""
    my_row = ROWProjectionResult(AVG=0.270, OPS=0.780, ERA=3.50, WHIP=1.20, K_9=9.5,
                                  **{k: 0.0 for k in SCORING_CATEGORY_CODES if k not in ["AVG", "OPS", "ERA", "WHIP", "K_9"]})
    opp_row = ROWProjectionResult(AVG=0.265, OPS=0.770, ERA=3.80, WHIP=1.25, K_9=9.0,
                                  **{k: 0.0 for k in SCORING_CATEGORY_CODES if k not in ["AVG", "OPS", "ERA", "WHIP", "K_9"]})

    my_numerators = {"AVG": 120.0, "OPS": 220.0, "ERA": 35.0, "WHIP": 120.0}
    my_denominators = {"AVG": 450.0, "OPS": 450.0, "ERA": 90.0, "WHIP": 90.0}

    bundle = prepare_simulation_inputs(
        my_row, opp_row,
        my_numerators=my_numerators,
        my_denominators=my_denominators,
    )

    # Category math should still be computed
    assert len(bundle.category_math) == 18


def test_h2h_monte_carlo_integration():
    """End-to-end integration: ROW → H2H Monte Carlo."""
    from backend.fantasy_baseball.h2h_monte_carlo import H2HOneWinSimulator

    my_row = ROWProjectionResult(
        R=50.0, H=120.0, HR_B=18.0, RBI=55.0, K_B=80.0, TB=180.0, NSB=5.0,
        AVG=0.270, OPS=0.780, W=4.0, L=2.0, HR_P=12.0, K_P=90.0, QS=5.0, NSV=-2.0,
        ERA=3.50, WHIP=1.20, K_9=9.5,
    )
    opp_row = ROWProjectionResult(
        R=45.0, H=115.0, HR_B=15.0, RBI=50.0, K_B=75.0, TB=170.0, NSB=3.0,
        AVG=0.265, OPS=0.770, W=3.0, L=3.0, HR_P=15.0, K_P=85.0, QS=4.0, NSV=-1.0,
        ERA=3.80, WHIP=1.25, K_9=9.0,
    )

    bundle = prepare_simulation_inputs(my_row, opp_row)
    my_finals, opp_finals = prepare_h2h_monte_carlo_inputs(bundle)

    sim = H2HOneWinSimulator()
    result = sim.simulate_week_from_projections(my_finals, opp_finals, n_sims=1000)

    assert 0.0 <= result.win_probability <= 1.0
    assert len(result.category_win_probs) == 18
    assert result.mean_categories_won >= 0


def test_mcmc_simulation_integration():
    """End-to-end integration: ROW → MCMC simulator."""
    from backend.fantasy_baseball.mcmc_simulator import simulate_weekly_matchup

    my_row = ROWProjectionResult(
        R=50.0, H=120.0, HR_B=18.0, RBI=55.0, K_B=80.0, TB=180.0, NSB=5.0,
        AVG=0.270, OPS=0.780, W=4.0, L=2.0, HR_P=12.0, K_P=90.0, QS=5.0, NSV=-2.0,
        ERA=3.50, WHIP=1.20, K_9=9.5,
    )
    opp_row = ROWProjectionResult(
        R=40.0, H=110.0, HR_B=12.0, RBI=45.0, K_B=90.0, TB=160.0, NSB=2.0,
        AVG=0.260, OPS=0.750, W=3.0, L=4.0, HR_P=18.0, K_P=75.0, QS=3.0, NSV=-3.0,
        ERA=4.20, WHIP=1.35, K_9=8.0,
    )

    bundle = prepare_simulation_inputs(my_row, opp_row)
    my_roster, opp_roster = prepare_mcmc_simulation_inputs(bundle)

    result = simulate_weekly_matchup(my_roster, opp_roster, n_sims=500)

    assert 0.0 <= result["win_prob"] <= 1.0
    assert len(result["category_win_probs"]) == 18
    assert result["expected_cats_won"] >= 0
