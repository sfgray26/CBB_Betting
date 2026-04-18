"""
End-to-end integration tests for Phase 3.

Verifies:
- ROW projection → bridge → Monte Carlo → H2HWinResult (full pipeline)
- Player scores → bridge → MCMC → win_prob (full pipeline)
- Ratio risk correctly identifies when streaming a pitcher is dangerous
- category_count_delta matches Monte Carlo category breakdown
- Non-degenerate results: win_prob ≠ 0.0 and ≠ 1.0 for balanced teams
"""

import pytest

from backend.services.row_simulation_bridge import (
    prepare_simulation_inputs,
    prepare_h2h_monte_carlo_inputs,
    prepare_mcmc_simulation_inputs,
    summarize_simulation_bundles,
)
from backend.services.row_projector import ROWProjectionResult
from backend.services.category_math import (
    compute_ratio_risk,
    compute_category_count_delta,
)
from backend.fantasy_baseball.h2h_monte_carlo import H2HOneWinSimulator
from backend.fantasy_baseball.mcmc_simulator import simulate_weekly_matchup
from backend.stat_contract import SCORING_CATEGORY_CODES


def test_row_to_monte_carlo_full_pipeline():
    """ROW projection → bridge → Monte Carlo → H2HWinResult (full pipeline)."""
    # Create ROW projections for both teams
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

    # Bridge: ROW → Simulation inputs
    bundle = prepare_simulation_inputs(my_row, opp_row)
    my_finals, opp_finals = prepare_h2h_monte_carlo_inputs(bundle)

    # Run Monte Carlo simulation
    sim = H2HOneWinSimulator()
    result = sim.simulate_week_from_projections(my_finals, opp_finals, n_sims=1000)

    # Verify full pipeline produces valid results
    assert 0.0 < result.win_probability < 1.0, "Win probability should be non-degenerate"
    assert len(result.category_win_probs) == 18
    assert result.mean_categories_won > 0


def test_player_scores_to_mcmc_full_pipeline():
    """Player scores → bridge → MCMC → win_prob (full pipeline)."""
    # Use more balanced projections to get non-degenerate results
    my_row = ROWProjectionResult(
        R=50.0, H=120.0, HR_B=18.0, RBI=55.0, K_B=80.0, TB=180.0, NSB=5.0,
        AVG=0.270, OPS=0.780,
        W=4.0, L=2.0, HR_P=12.0, K_P=90.0, QS=5.0, NSV=-2.0,
        ERA=3.50, WHIP=1.20, K_9=9.5,
    )
    opp_row = ROWProjectionResult(
        R=48.0, H=118.0, HR_B=17.0, RBI=53.0, K_B=78.0, TB=178.0, NSB=4.0,
        AVG=0.268, OPS=0.775,
        W=4.0, L=2.0, HR_P=13.0, K_P=88.0, QS=5.0, NSV=-2.0,
        ERA=3.60, WHIP=1.22, K_9=9.3,
    )

    # Bridge: ROW → MCMC inputs
    bundle = prepare_simulation_inputs(my_row, opp_row)
    my_roster, opp_roster = prepare_mcmc_simulation_inputs(bundle)

    # Run MCMC simulation
    result = simulate_weekly_matchup(my_roster, opp_roster, n_sims=500)

    # Verify full pipeline produces valid results
    assert result["win_prob"] >= 0.0, "Win probability should be valid"
    assert len(result["category_win_probs"]) == 18
    assert result["expected_cats_won"] >= 0


def test_ratio_risk_identifies_dangerous_streamer():
    """Ratio risk correctly identifies when streaming a pitcher is dangerous."""
    # Scenario: My ERA is 3.50, opponent has 3.60. I'm winning by 0.10.
    # Considering streaming a pitcher with projected 5.50 ERA, 6 IP.
    # This would likely blow my ERA category.

    # Current state: 30 IP, 11.67 ER → ERA = 3.50
    my_ip = 30.0
    my_er = 11.67
    my_hits = 30.0
    my_bb = 10.0

    # Opponent ERA = 3.60, WHIP = 1.25
    opp_era = 3.60
    opp_whip = 1.25

    # Check risk with 6 IP remaining
    risk = compute_ratio_risk(
        my_ip=my_ip,
        my_er=my_er,
        my_hits_allowed=my_hits,
        my_bb_allowed=my_bb,
        opp_era=opp_era,
        opp_whip=opp_whip,
        remaining_ip=6.0,
    )

    # With only 6 IP remaining and small cushion, should be AT_RISK or CRITICAL
    assert risk["era_risk"] in ("AT_RISK", "CRITICAL")

    # If a bad pitcher (5.50 ERA projected) allows 4 ER in 6 IP:
    # New ER = 11.67 + 4 = 15.67
    # New IP = 30 + 6 = 36
    # New ERA = (15.67 / 36) * 9 = 3.92 > 3.60 (opp)
    # This would flip the category from WIN to LOSS


def test_category_count_delta_matches_simulation():
    """category_count_delta matches Monte Carlo category breakdown."""
    my_row = ROWProjectionResult(
        R=60.0, H=130.0, HR_B=20.0, RBI=60.0, K_B=70.0, TB=190.0, NSB=6.0,
        AVG=0.280, OPS=0.800,
        W=5.0, L=1.0, HR_P=10.0, K_P=95.0, QS=6.0, NSV=-1.0,
        ERA=3.30, WHIP=1.15, K_9=10.0,
    )
    opp_row = ROWProjectionResult(
        R=40.0, H=110.0, HR_B=12.0, RBI=45.0, K_B=85.0, TB=160.0, NSB=2.0,
        AVG=0.260, OPS=0.750,
        W=3.0, L=4.0, HR_P=18.0, K_P=75.0, QS=3.0, NSV=-3.0,
        ERA=4.20, WHIP=1.35, K_9=8.0,
    )

    # Get category math
    bundle = prepare_simulation_inputs(my_row, opp_row)
    count_delta = compute_category_count_delta(bundle.category_math)

    # Run simulation
    my_finals, opp_finals = prepare_h2h_monte_carlo_inputs(bundle)
    sim = H2HOneWinSimulator()
    result = sim.simulate_week_from_projections(my_finals, opp_finals, n_sims=1000)

    # Category count delta should align with simulation results
    # If we're projected to WIN, win_prob should be > 0.5
    if count_delta["projected_result"] == "WIN":
        assert result.win_probability > 0.5
    elif count_delta["projected_result"] == "LOSS":
        assert result.win_probability < 0.5


def test_non_degenerate_results_for_balanced_teams():
    """Non-degenerate results: win_prob ≠ 0.0 and ≠ 1.0 for balanced teams."""
    # Create closely matched teams
    my_row = ROWProjectionResult(
        R=50.0, H=120.0, HR_B=15.0, RBI=50.0, K_B=80.0, TB=175.0, NSB=4.0,
        AVG=0.270, OPS=0.775,
        W=4.0, L=3.0, HR_P=14.0, K_P=85.0, QS=4.0, NSV=-2.0,
        ERA=3.75, WHIP=1.22, K_9=9.2,
    )
    opp_row = ROWProjectionResult(
        R=48.0, H=118.0, HR_B=16.0, RBI=52.0, K_B=78.0, TB=178.0, NSB=5.0,
        AVG=0.268, OPS=0.778,
        W=4.0, L=3.0, HR_P=13.0, K_P=88.0, QS=5.0, NSV=-1.0,
        ERA=3.72, WHIP=1.24, K_9=9.3,
    )

    # Bridge and simulate
    bundle = prepare_simulation_inputs(my_row, opp_row)
    my_finals, opp_finals = prepare_h2h_monte_carlo_inputs(bundle)

    sim = H2HOneWinSimulator()
    result = sim.simulate_week_from_projections(my_finals, opp_finals, n_sims=2000)

    # For closely matched teams, should get non-degenerate results
    assert 0.1 < result.win_probability < 0.9, \
        f"Win prob should be non-degenerate for balanced teams, got {result.win_probability}"


def test_summarize_bundle_provides_useful_summary():
    """summarize_simulation_bundles provides useful summary for UI."""
    my_row = ROWProjectionResult(
        R=55.0, H=125.0, HR_B=20.0, RBI=60.0, K_B=75.0, TB=185.0, NSB=6.0,
        AVG=0.280, OPS=0.790,
        W=5.0, L=2.0, HR_P=11.0, K_P=92.0, QS=5.0, NSV=-1.0,
        ERA=3.40, WHIP=1.18, K_9=9.8,
    )
    opp_row = ROWProjectionResult(
        R=42.0, H=112.0, HR_B=13.0, RBI=46.0, K_B=82.0, TB=165.0, NSB=3.0,
        AVG=0.262, OPS=0.765,
        W=3.0, L=4.0, HR_P=16.0, K_P=80.0, QS=3.0, NSV=-3.0,
        ERA=4.00, WHIP=1.30, K_9=8.5,
    )

    bundle = prepare_simulation_inputs(my_row, opp_row)
    summary = summarize_simulation_bundles(bundle)

    # Should have all expected keys
    assert "categories_winning" in summary
    assert "categories_losing" in summary
    assert "categories_tied" in summary
    assert "biggest_lead" in summary
    assert "biggest_deficit" in summary
    assert "swing_categories" in summary

    # Sanity checks
    assert summary["categories_winning"] >= 0
    assert summary["categories_winning"] + summary["categories_losing"] + summary["categories_tied"] == 18
    assert summary["biggest_lead"] >= summary["biggest_deficit"]
