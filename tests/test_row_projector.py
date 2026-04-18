"""
Tests for row_projector.py -- P13 Rest-of-Week projections.

Pure function tests only -- zero I/O, zero DB, zero mocks needed.
"""

import pytest

from backend.services.row_projector import (
    ROWProjectionResult,
    compute_row_projection,
    estimate_hitter_games_remaining,
    estimate_pitcher_games_remaining,
)


# ===========================================================================
# Counting stats: basic projection
# ===========================================================================

def test_single_hitter_runs_projection():
    """One hitter with 14 runs over 14 days, 1 game remaining -> projects 1 run."""
    rolling = {"player_123": {"w_runs": 14.0}}  # 1 run/day average
    games = {"player_123": 1}

    result = compute_row_projection(rolling, games_remaining=games)

    # Daily rate = 14/14 = 1.0, blended = 0.6*1.0 + 0.4*0 = 0.6
    # But with no season data, season_rate = 0, so blended = 0.6 * 1 = 0.6
    # Projected = 0.6 * 1 game = 0.6
    assert result.R == pytest.approx(0.6, rel=1e-6)


def test_single_hitter_multiple_games():
    """Hitter with 1 run/day, 3 games remaining -> projects 1.8 runs."""
    rolling = {"player_123": {"w_runs": 14.0}}
    games = {"player_123": 3}

    result = compute_row_projection(rolling, games_remaining=games)

    # Daily rate = 1.0, blended = 0.6 (no season data)
    # Projected = 0.6 * 3 = 1.8
    assert result.R == pytest.approx(1.8, rel=1e-6)


def test_multiple_hitters_accumulate():
    """Two hitters both score runs; projections accumulate."""
    rolling = {
        "player_123": {"w_runs": 14.0},  # 1 run/day
        "player_456": {"w_runs": 28.0},  # 2 runs/day
    }
    games = {"player_123": 2, "player_456": 2}

    result = compute_row_projection(rolling, games_remaining=games)

    # player_123: 0.6 * 2 = 1.2
    # player_456: daily = 28/14 = 2, blended = 0.6*2 = 1.2, proj = 1.2*2 = 2.4
    # Total = 1.2 + 2.4 = 3.6
    assert result.R == pytest.approx(3.6, rel=1e-6)


def test_hitter_hits_projection():
    """Hits accumulate like other counting stats."""
    rolling = {"player_123": {"w_hits": 28.0}}  # 2 hits/day
    games = {"player_123": 2}

    result = compute_row_projection(rolling, games_remaining=games)

    # Daily = 2, blended = 0.6*2 = 1.2, proj = 1.2*2 = 2.4
    assert result.H == pytest.approx(2.4, rel=1e-6)


def test_hitter_hr_projection():
    """Home runs accumulate like other counting stats."""
    rolling = {"player_123": {"w_home_runs": 7.0}}  # 0.5 HR/day
    games = {"player_123": 2}

    result = compute_row_projection(rolling, games_remaining=games)

    # Daily = 0.5, blended = 0.6*0.5 = 0.3, proj = 0.3*2 = 0.6
    assert result.HR_B == pytest.approx(0.6, rel=1e-6)


def test_hitter_rbi_projection():
    """RBI accumulates like other counting stats."""
    rolling = {"player_123": {"w_rbi": 21.0}}  # 1.5 RBI/day
    games = {"player_123": 2}

    result = compute_row_projection(rolling, games_remaining=games)

    # Daily = 1.5, blended = 0.6*1.5 = 0.9, proj = 0.9*2 = 1.8
    assert result.RBI == pytest.approx(1.8, rel=1e-6)


def test_hitter_strikeouts_projection():
    """Batter strikeouts accumulate (lower-is-better, but projection is sum)."""
    rolling = {"player_123": {"w_strikeouts_bat": 14.0}}  # 1 K/day
    games = {"player_123": 2}

    result = compute_row_projection(rolling, games_remaining=games)

    # Daily = 1, blended = 0.6, proj = 0.6*2 = 1.2
    assert result.K_B == pytest.approx(1.2, rel=1e-6)


def test_hitter_tb_projection():
    """Total bases accumulate like other counting stats."""
    rolling = {"player_123": {"w_tb": 21.0}}  # 1.5 TB/day
    games = {"player_123": 2}

    result = compute_row_projection(rolling, games_remaining=games)

    # Daily = 1.5, blended = 0.6*1.5 = 0.9, proj = 0.9*2 = 1.8
    assert result.TB == pytest.approx(1.8, rel=1e-6)


def test_nsb_projection():
    """Net stolen bases (SB - CS) accumulate."""
    rolling = {"player_123": {"w_net_stolen_bases": 3.5}}  # 0.25 NSB/day
    games = {"player_123": 2}

    result = compute_row_projection(rolling, games_remaining=games)

    # Daily = 0.25, blended = 0.6*0.25 = 0.15, proj = 0.15*2 = 0.3
    assert result.NSB == pytest.approx(0.3, rel=1e-6)


# ===========================================================================
# Pitching counting stats
# ===========================================================================

def test_pitcher_strikeouts_projection():
    """Pitcher strikeouts accumulate."""
    rolling = {"player_789": {"w_strikeouts_pit": 42.0}}  # 3 K/day
    games = {"player_789": 1}

    result = compute_row_projection(rolling, games_remaining=games)

    # Daily = 3, blended = 0.6*3 = 1.8, proj = 1.8*1 = 1.8
    assert result.K_P == pytest.approx(1.8, rel=1e-6)


def test_pitcher_qs_projection():
    """Quality starts accumulate (not ratio-based)."""
    rolling = {"player_789": {"w_qs": 1.0}}  # 1 QS over 14 days
    games = {"player_789": 1}

    result = compute_row_projection(rolling, games_remaining=games)

    # Daily = 1/14 ≈ 0.0714, blended = 0.6*0.0714 = 0.0429, proj = 0.0429
    assert result.QS == pytest.approx(0.0429, rel=1e-3)


# ===========================================================================
# Greenfield categories (no upstream data)
# ===========================================================================

def test_greenfield_categories_return_zero():
    """W, L, HR_P, NSV have no upstream data and return 0.0."""
    rolling = {"player_123": {"w_runs": 14.0}}
    games = {"player_123": 1}

    result = compute_row_projection(rolling, games_remaining=games)

    assert result.W == 0.0
    assert result.L == 0.0
    assert result.HR_P == 0.0
    assert result.NSV == 0.0


# ===========================================================================
# Ratio stats: AVG
# ===========================================================================

def test_avg_single_hitter():
    """Team AVG = sum(H) / sum(AB) for one hitter."""
    rolling = {"player_123": {"w_hits": 28.0, "w_ab": 100.0}}  # .280 avg
    games = {"player_123": 1}

    result = compute_row_projection(rolling, games_remaining=games)

    # H_daily = 28/14 * 0.6 = 1.2, AB_daily = 100/14 * 0.6 = 4.286
    # AVG = 1.2 / 4.286 = 0.28
    assert result.AVG == pytest.approx(0.28, rel=0.01)


def test_avg_two_hitters():
    """Team AVG is NOT the average of player AVGs."""
    rolling = {
        "player_123": {"w_hits": 28.0, "w_ab": 100.0},  # .280 avg
        "player_456": {"w_hits": 35.0, "w_ab": 100.0},  # .350 avg
    }
    games = {"player_123": 1, "player_456": 1}

    result = compute_row_projection(rolling, games_remaining=games)

    # H_daily: 28/14*0.6 = 1.2, 35/14*0.6 = 1.5 -> sum = 2.7
    # AB_daily: 100/14*0.6 = 4.286, 100/14*0.6 = 4.286 -> sum = 8.571
    # AVG = 2.7 / 8.571 = 0.315
    assert result.AVG == pytest.approx(0.315, rel=0.01)


def test_avg_zero_ab_returns_zero():
    """If no AB, AVG returns 0 (not division by zero)."""
    rolling = {"player_123": {"w_hits": 0.0, "w_ab": 0.0}}
    games = {"player_123": 1}

    result = compute_row_projection(rolling, games_remaining=games)

    assert result.AVG == 0.0


# ===========================================================================
# Ratio stats: OPS
# ===========================================================================

def test_ops_single_hitter():
    """OPS = OBP + SLG."""
    rolling = {"player_123": {"w_hits": 28.0, "w_ab": 100.0, "w_walks": 14.0, "w_tb": 42.0}}
    games = {"player_123": 1}

    result = compute_row_projection(rolling, games_remaining=games)

    # H_daily = 1.2, AB_daily = 4.286, BB_daily = 0.6, TB_daily = 1.8
    # OBP = (1.2 + 0.6) / (4.286 + 0.6) = 1.8 / 4.886 = 0.368
    # SLG = 1.8 / 4.286 = 0.42
    # OPS = 0.368 + 0.42 = 0.788
    assert result.OPS == pytest.approx(0.788, rel=0.01)


# ===========================================================================
# Ratio stats: ERA
# ===========================================================================

def test_era_single_pitcher():
    """ERA = 27 * sum(ER) / sum(IP_outs)."""
    # IP_outs = IP * 3. For 7 IP, that's 21 outs
    rolling = {"player_789": {"w_earned_runs": 14.0, "w_ip_outs": 21.0}}  # 7 IP, 14 ER -> 6.00 ERA
    games = {"player_789": 1}

    result = compute_row_projection(rolling, games_remaining=games)

    # ER_daily = 14/14 * 0.6 = 0.6, IP_outs_daily = 21/14 * 0.6 = 0.9
    # ERA = 27 * 0.6 / 0.9 = 18.0 (projected ERA for that one game)
    # Note: This is the per-game ERA contribution, not a cumulative stat
    assert result.ERA == pytest.approx(18.0, rel=0.01)


def test_era_two_pitchers():
    """Team ERA blends both pitchers."""
    rolling = {
        "p1": {"w_earned_runs": 7.0, "w_ip_outs": 21.0},   # 3.00 ERA
        "p2": {"w_earned_runs": 14.0, "w_ip_outs": 21.0},  # 6.00 ERA
    }
    games = {"p1": 1, "p2": 1}

    result = compute_row_projection(rolling, games_remaining=games)

    # p1: ER_daily = 0.3, IP_outs_daily = 0.9
    # p2: ER_daily = 0.6, IP_outs_daily = 0.9
    # ERA = 27 * (0.3 + 0.6) / (0.9 + 0.9) = 27 * 0.9 / 1.8 = 13.5
    assert result.ERA == pytest.approx(13.5, rel=0.01)


def test_era_zero_ip_returns_zero():
    """If no IP, ERA returns 0 (not division by zero)."""
    rolling = {"player_789": {"w_earned_runs": 0.0, "w_ip_outs": 0.0}}
    games = {"player_789": 1}

    result = compute_row_projection(rolling, games_remaining=games)

    assert result.ERA == 0.0


# ===========================================================================
# Ratio stats: WHIP
# ===========================================================================

def test_whip_single_pitcher():
    """WHIP = 3 * sum(H + BB) / sum(IP_outs)."""
    rolling = {
        "player_789": {
            "w_hits_allowed": 21.0,
            "w_walks_allowed": 7.0,
            "w_ip_outs": 21.0,  # 7 IP
        }
    }
    games = {"player_789": 1}

    result = compute_row_projection(rolling, games_remaining=games)

    # H_daily = 21/14 * 0.6 = 0.9
    # BB_daily = 7/14 * 0.6 = 0.3
    # IP_outs_daily = 21/14 * 0.6 = 0.9
    # WHIP = 3 * (0.9 + 0.3) / 0.9 = 3 * 1.2 / 0.9 = 4.0
    assert result.WHIP == pytest.approx(4.0, rel=0.01)


# ===========================================================================
# Ratio stats: K/9
# ===========================================================================

def test_k9_single_pitcher():
    """K/9 = 27 * sum(K) / sum(IP_outs)."""
    rolling = {"player_789": {"w_strikeouts_pit": 42.0, "w_ip_outs": 21.0}}  # 18 K/9
    games = {"player_789": 1}

    result = compute_row_projection(rolling, games_remaining=games)

    # K_daily = 42/14 * 0.6 = 1.8
    # IP_outs_daily = 21/14 * 0.6 = 0.9
    # K/9 = 27 * 1.8 / 0.9 = 54
    assert result.K_9 == pytest.approx(54.0, rel=0.01)


# ===========================================================================
# Blended rate with season data
# ===========================================================================

def test_blended_rate_rolling_and_season():
    """Blended rate = 60% rolling + 40% season."""
    rolling = {"player_123": {"w_runs": 14.0}}  # 1 run/day rolling
    season = {"player_123": {"runs": 150.0}}    # 1.5 runs/day season
    games = {"player_123": 1}

    result = compute_row_projection(
        rolling,
        season_stats_by_player=season,
        games_remaining=games,
    )

    # Rolling daily = 1.0, Season daily = 150/100 = 1.5
    # Blended = 0.6*1.0 + 0.4*1.5 = 0.6 + 0.6 = 1.2
    assert result.R == pytest.approx(1.2, rel=1e-6)


def test_custom_weights():
    """Custom rolling/season weights override defaults."""
    rolling = {"player_123": {"w_runs": 14.0}}
    season = {"player_123": {"runs": 100.0}}  # 1 run/day
    games = {"player_123": 1}

    result = compute_row_projection(
        rolling,
        season_stats_by_player=season,
        games_remaining=games,
        rolling_weight=0.8,  # Heavier rolling weight
        season_weight=0.2,
    )

    # Blended = 0.8*1.0 + 0.2*1.0 = 1.0
    assert result.R == pytest.approx(1.0, rel=1e-6)


# ===========================================================================
# Edge cases
# ===========================================================================

def test_player_with_zero_games_remaining_skipped():
    """Players with 0 games remaining don't contribute."""
    rolling = {
        "player_123": {"w_runs": 14.0},
        "player_456": {"w_runs": 28.0},
    }
    games = {"player_123": 0, "player_456": 2}

    result = compute_row_projection(rolling, games_remaining=games)

    # Only player_456 contributes
    # Daily = 2, blended = 0.6*2 = 1.2, proj = 1.2*2 = 2.4
    assert result.R == pytest.approx(2.4, rel=1e-6)


def test_empty_roster_returns_zeros():
    """Empty roster produces all-zero projections."""
    result = compute_row_projection({}, games_remaining={})

    assert result.R == 0.0
    assert result.H == 0.0
    assert result.HR_B == 0.0
    assert result.AVG == 0.0
    assert result.ERA == 0.0


def test_to_dict_conversion():
    """to_dict() produces flat dict with all 18 categories."""
    result = ROWProjectionResult(
        R=10.0, H=25.0, HR_B=5.0, RBI=20.0, K_B=30.0, TB=40.0, NSB=3.0,
        AVG=0.280, OPS=0.850,
        W=3.0, L=2.0, HR_P=8.0, K_P=60.0, QS=4.0, NSV=-1.0,
        ERA=3.50, WHIP=1.20, K_9=10.5,
    )

    d = result.to_dict()

    assert len(d) == 18
    assert d["R"] == 10.0
    assert d["AVG"] == 0.280
    assert d["ERA"] == 3.50


# ===========================================================================
# Games remaining estimation
# ===========================================================================

def test_healthy_hitter_games_remaining():
    """Healthy hitter gets all remaining days."""
    games = estimate_hitter_games_remaining(
        current_day_of_week=3,  # Thursday
        days_remaining_in_week=4,  # Thu-Sun
        player_status="healthy",
    )
    assert games == 4


def test_injured_hitter_zero_games():
    """Injured hitter gets 0 games."""
    games = estimate_hitter_games_remaining(
        current_day_of_week=3,
        days_remaining_in_week=4,
        player_status="out",
    )
    assert games == 0


def test_dl_hitter_zero_games():
    """DL-listed hitter gets 0 games."""
    games = estimate_hitter_games_remaining(
        current_day_of_week=3,
        days_remaining_in_week=4,
        player_status="dl",
    )
    assert games == 0


def test_probable_pitcher_gets_one_start():
    """Pitcher listed as probable gets 1 start."""
    probable = {"NYY": 12345}  # ID doesn't matter, just presence
    games = estimate_pitcher_games_remaining(
        current_day_of_week=3,
        days_remaining_in_week=4,
        probable_starters=probable,
        pitcher_team="NYY",
    )
    assert games == 1


def test_non_probable_pitcher_zero_starts():
    """Pitcher not listed as probable gets 0 starts (until rotation math implemented)."""
    probable = {"BOS": 99999}
    games = estimate_pitcher_games_remaining(
        current_day_of_week=3,
        days_remaining_in_week=4,
        probable_starters=probable,
        pitcher_team="NYY",  # Not in probables
    )
    assert games == 0
