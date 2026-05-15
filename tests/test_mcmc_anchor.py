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
