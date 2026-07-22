"""Tests for current-stats anchoring in MCMC simulator."""
import pytest
from backend.fantasy_baseball.mcmc_simulator import simulate_weekly_matchup


def _make_player(name: str, hr: float = 0.0) -> dict:
    return {
        "name": name,
        "positions": ["OF"],
        "starts_this_week": 0,
        "cat_scores": {"hr_b": hr},
    }


def test_anchor_preserves_current_lead():
    """If I'm up 8 HRs today and remaining_fraction=0.3, win_prob for HR_B must be > 0.80."""
    my_roster = [_make_player("A", hr=0.5), _make_player("B", hr=0.3)]
    opp_roster = [_make_player("C", hr=0.5), _make_player("D", hr=0.3)]

    result = simulate_weekly_matchup(
        my_roster, opp_roster,
        categories=["hr_b"],
        my_current_stats={"hr_b": 8.0},
        opp_current_stats={"hr_b": 0.0},
        remaining_fraction=0.3,
        n_sims=2000,
        seed=42,
    )
    wp = result["category_win_probs"]["hr_b"]
    assert wp > 0.80, f"Expected >0.80 win prob for HR with 8-0 lead, got {wp}"


def test_no_anchor_returns_near_fifty():
    """Without anchor, equal rosters should produce ~0.50 win prob."""
    roster = [_make_player("A", hr=0.5)]
    result = simulate_weekly_matchup(
        roster, roster,
        categories=["hr_b"],
        n_sims=2000,
        seed=42,
    )
    wp = result["category_win_probs"]["hr_b"]
    assert 0.35 <= wp <= 0.65, f"Expected ~0.50 for equal rosters, got {wp}"


def test_anchor_does_not_change_api_when_no_current_stats():
    """Calling without current_stats must return same keys as before (backward compat)."""
    roster = [_make_player("A", hr=0.5)]
    result = simulate_weekly_matchup(roster, roster, categories=["hr_b"], n_sims=200, seed=1)
    assert "win_prob" in result
    assert "category_projections" in result
    assert "data_quality" in result


def test_data_quality_flag_degraded():
    """If >30% of roster has empty cat_scores, data_quality must be 'degraded'."""
    # 3 players, 2 with empty cat_scores → 33% coverage → degraded
    roster_my = [
        _make_player("A", hr=0.5),
        {"name": "B", "positions": ["OF"], "starts_this_week": 0, "cat_scores": {}},
        {"name": "C", "positions": ["OF"], "starts_this_week": 0, "cat_scores": {}},
    ]
    roster_opp = [_make_player("X", hr=0.5), _make_player("Y", hr=0.4), _make_player("Z", hr=0.3)]

    result = simulate_weekly_matchup(roster_my, roster_opp, categories=["hr_b"], n_sims=200, seed=1)
    assert result["data_quality"] == "degraded"


def test_data_quality_flag_ok():
    """If all roster players have non-zero cat_scores, data_quality must be 'ok'."""
    roster = [_make_player(f"P{i}", hr=0.5) for i in range(5)]
    result = simulate_weekly_matchup(roster, roster, categories=["hr_b"], n_sims=200, seed=1)
    assert result["data_quality"] == "ok"


def _make_pitcher(name: str, l_score: float = 0.0, k_b_score: float = 0.0) -> dict:
    return {
        "name": name,
        "positions": ["SP"],
        "starts_this_week": 1,
        "cat_scores": {"l": l_score, "k_b": k_b_score},
    }


def test_anchor_lower_is_better_losses_lead_preserved():
    """Regression (UAT 2026-07-22): ahead 1-4 in L mid-week must score as a WIN.

    Raw scoreboard anchors for LOWER_IS_BETTER cats were previously added
    without sign inversion, so the team with FEWER losses got the smaller
    total and lost every sim (scored BEHIND + PUNT? in War Room).
    """
    my_roster = [_make_pitcher("A"), _make_pitcher("B")]
    opp_roster = [_make_pitcher("C"), _make_pitcher("D")]

    result = simulate_weekly_matchup(
        my_roster, opp_roster,
        categories=["l"],
        my_current_stats={"l": 1.0},
        opp_current_stats={"l": 4.0},
        remaining_fraction=0.3,
        n_sims=2000,
        seed=42,
    )
    wp = result["category_win_probs"]["l"]
    assert wp > 0.80, f"Expected >0.80 win prob for L with 1-vs-4 lead, got {wp}"


def test_anchor_lower_is_better_batter_ks_lead_preserved():
    """Regression (UAT 2026-07-22): up 22-35 in K_B (fewer Ks) must score as a WIN.

    K_B is also in _COUNT_CATEGORIES — the zero-clamp must not erase the
    sign-inverted anchor.
    """
    my_roster = [_make_pitcher("A"), _make_pitcher("B")]
    opp_roster = [_make_pitcher("C"), _make_pitcher("D")]

    result = simulate_weekly_matchup(
        my_roster, opp_roster,
        categories=["k_b"],
        my_current_stats={"k_b": 22.0},
        opp_current_stats={"k_b": 35.0},
        remaining_fraction=0.3,
        n_sims=2000,
        seed=42,
    )
    wp = result["category_win_probs"]["k_b"]
    assert wp > 0.80, f"Expected >0.80 win prob for K_B with 22-vs-35 lead, got {wp}"


def test_lower_is_better_display_projections_positive():
    """my_proj/opp_proj for lower-is-better cats must display as real stat values."""
    my_roster = [_make_pitcher("A")]
    opp_roster = [_make_pitcher("B")]

    result = simulate_weekly_matchup(
        my_roster, opp_roster,
        categories=["l"],
        my_current_stats={"l": 1.0},
        opp_current_stats={"l": 4.0},
        remaining_fraction=0.3,
        n_sims=500,
        seed=7,
    )
    proj = result["category_projections"][0]
    assert proj["category"] == "L"
    assert proj["my_proj"] > 0, f"my_proj should display ≈ current losses, got {proj['my_proj']}"
    assert proj["opp_proj"] > proj["my_proj"], (
        f"opp (4 L) should display higher than mine (1 L): {proj}"
    )
