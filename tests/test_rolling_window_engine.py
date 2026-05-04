"""
Tests for rolling_window_engine.py -- P13 Derived Stats.

Pure function tests only -- zero I/O, zero DB, zero mocks needed.
ORM rows are simulated via SimpleNamespace objects with the same attributes
as the MLBPlayerStats ORM class.
"""

import math
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from backend.services.rolling_window_engine import (
    RollingWindowResult,
    compute_all_rolling_windows,
    compute_rolling_window,
    parse_ip,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stat_row(
    bdl_player_id: int,
    game_date: date,
    # Batting
    ab=None, hits=None, doubles=None, triples=None, home_runs=None,
    rbi=None, runs=None, walks=None, strikeouts_bat=None, stolen_bases=None,
    # Pitching
    innings_pitched=None, hits_allowed=None, runs_allowed=None,
    earned_runs=None, walks_allowed=None, strikeouts_pit=None,
):
    """Construct a minimal ORM-like stat row."""
    return SimpleNamespace(
        bdl_player_id=bdl_player_id,
        game_date=game_date,
        ab=ab,
        hits=hits,
        doubles=doubles,
        triples=triples,
        home_runs=home_runs,
        rbi=rbi,
        runs=runs,
        walks=walks,
        strikeouts_bat=strikeouts_bat,
        stolen_bases=stolen_bases,
        innings_pitched=innings_pitched,
        hits_allowed=hits_allowed,
        runs_allowed=runs_allowed,
        earned_runs=earned_runs,
        walks_allowed=walks_allowed,
        strikeouts_pit=strikeouts_pit,
    )


TODAY = date(2026, 4, 6)


# ===========================================================================
# IP Parser tests
# ===========================================================================

def test_parse_ip_none_returns_none():
    assert parse_ip(None) is None


def test_parse_ip_whole_innings():
    assert parse_ip("9") == 9.0


def test_parse_ip_zero_outs():
    assert parse_ip("0.0") == 0.0


def test_parse_ip_one_out():
    result = parse_ip("0.1")
    assert result == pytest.approx(1 / 3, rel=1e-6)


def test_parse_ip_two_outs():
    result = parse_ip("6.2")
    assert result == pytest.approx(6 + 2 / 3, rel=1e-6)


def test_parse_ip_full_game():
    assert parse_ip("9.0") == pytest.approx(9.0)


def test_parse_ip_empty_string_returns_none():
    assert parse_ip("") is None


def test_parse_ip_one_inning_one_out():
    result = parse_ip("1.1")
    assert result == pytest.approx(1 + 1 / 3, rel=1e-6)


# ===========================================================================
# Single game -- hitter
# ===========================================================================

def test_single_game_hitter_7d():
    """1 game, all batting fields populated, pitching fields None."""
    row = _stat_row(
        bdl_player_id=100,
        game_date=TODAY,
        ab=4, hits=2, doubles=1, triples=0, home_runs=0,
        rbi=1, walks=1, strikeouts_bat=1, stolen_bases=0,
    )
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)

    assert result is not None
    assert result.bdl_player_id == 100
    assert result.window_days == 7
    assert result.games_in_window == 1

    # Weight for today = 0.95 ** 0 = 1.0
    assert result.w_ab == pytest.approx(4.0)
    assert result.w_hits == pytest.approx(2.0)
    assert result.w_doubles == pytest.approx(1.0)
    assert result.w_home_runs == pytest.approx(0.0)
    assert result.w_walks == pytest.approx(1.0)

    # Batting rates
    assert result.w_avg == pytest.approx(2 / 4)
    # OBP: (hits + walks) / (ab + walks) = 3 / 5
    assert result.w_obp == pytest.approx(3 / 5)
    # SLG: singles=1, doubles=1, TB=1+2=3, slg=3/4
    assert result.w_slg == pytest.approx(3 / 4)
    assert result.w_ops == pytest.approx(result.w_obp + result.w_slg)

    # Pitching fields are all None
    assert result.w_ip is None
    assert result.w_era is None
    assert result.w_whip is None
    assert result.w_k_per_9 is None


# ===========================================================================
# Single game -- pitcher
# ===========================================================================

def test_single_game_pitcher_7d():
    """1 pitching game, batting fields None, pitching fields populated."""
    row = _stat_row(
        bdl_player_id=200,
        game_date=TODAY,
        innings_pitched="7.0",
        earned_runs=2,
        hits_allowed=5,
        walks_allowed=1,
        strikeouts_pit=8,
    )
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)

    assert result is not None
    assert result.games_in_window == 1

    assert result.w_ip == pytest.approx(7.0)
    assert result.w_earned_runs == pytest.approx(2.0)
    assert result.w_hits_allowed == pytest.approx(5.0)
    assert result.w_walks_allowed == pytest.approx(1.0)
    assert result.w_strikeouts_pit == pytest.approx(8.0)

    # Derived pitching rates
    assert result.w_era == pytest.approx(9 * 2 / 7)
    assert result.w_whip == pytest.approx((5 + 1) / 7)
    assert result.w_k_per_9 == pytest.approx(9 * 8 / 7)

    # Batting fields None (no ab)
    assert result.w_ab is None
    assert result.w_avg is None


# ===========================================================================
# Decay weighting
# ===========================================================================

def test_two_games_decay_applied():
    """Two games: today and 3 days ago. Verify 3-day-old game gets 0.95^3 weight."""
    today_row = _stat_row(100, TODAY, ab=4, hits=2)
    old_row = _stat_row(100, TODAY - timedelta(days=3), ab=3, hits=1)

    result = compute_rolling_window([today_row, old_row], as_of_date=TODAY, window_days=7)

    assert result is not None
    assert result.games_in_window == 2

    w_today = 0.95 ** 0   # = 1.0
    w_old = 0.95 ** 3     # ~= 0.857375

    expected_w_ab = w_today * 4 + w_old * 3
    expected_w_hits = w_today * 2 + w_old * 1

    assert result.w_ab == pytest.approx(expected_w_ab, rel=1e-6)
    assert result.w_hits == pytest.approx(expected_w_hits, rel=1e-6)


def test_avg_computed_from_weighted_sums():
    """w_avg must equal w_hits / w_ab, not a weighted average of per-game .avg."""
    # Game 1 (today):        4 AB, 2 H  -> raw avg = 0.500
    # Game 2 (yesterday):    3 AB, 1 H  -> raw avg = 0.333
    # Decay weights:         w1=1.0, w2=0.95
    # w_ab   = 1.0*4 + 0.95*3 = 6.85
    # w_hits = 1.0*2 + 0.95*1 = 2.95
    # w_avg  = 2.95 / 6.85
    row1 = _stat_row(100, TODAY, ab=4, hits=2)
    row2 = _stat_row(100, TODAY - timedelta(days=1), ab=3, hits=1)

    result = compute_rolling_window([row1, row2], as_of_date=TODAY, window_days=7)

    w1, w2 = 1.0, 0.95
    expected_avg = (w1 * 2 + w2 * 1) / (w1 * 4 + w2 * 3)
    assert result.w_avg == pytest.approx(expected_avg, rel=1e-6)


# ===========================================================================
# Total bases
# ===========================================================================

def test_slg_uses_total_bases():
    """HR=1 counts 4 TB, triple=1 counts 3, double=1 counts 2, single counts 1."""
    # 8 AB: 1 HR (4TB) + 1 triple (3TB) + 1 double (2TB) + 1 single (1TB) = 10 TB
    # slg = 10 / 8 = 1.25
    row = _stat_row(
        100, TODAY,
        ab=8, hits=4, doubles=1, triples=1, home_runs=1,
    )
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)

    assert result.w_slg == pytest.approx(10 / 8)


# ===========================================================================
# OBP
# ===========================================================================

def test_obp_approximate_formula():
    """w_obp = (w_hits + w_walks) / (w_ab + w_walks). No HBP/SF."""
    row = _stat_row(100, TODAY, ab=5, hits=2, walks=1)
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)

    # OBP = (2 + 1) / (5 + 1) = 0.5
    assert result.w_obp == pytest.approx(3 / 6)


# ===========================================================================
# Pitching derived rates
# ===========================================================================

def test_era_derived_from_er_and_ip():
    """w_era = 9 * w_earned_runs / w_ip."""
    row = _stat_row(200, TODAY, innings_pitched="6.0", earned_runs=3)
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)

    assert result.w_era == pytest.approx(9 * 3 / 6)


def test_whip_derived_from_h_bb_ip():
    """w_whip = (w_hits_allowed + w_walks_allowed) / w_ip."""
    row = _stat_row(200, TODAY, innings_pitched="6.2", hits_allowed=7, walks_allowed=2)
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)

    expected_ip = parse_ip("6.2")  # 6.667
    assert result.w_whip == pytest.approx((7 + 2) / expected_ip, rel=1e-5)


# ===========================================================================
# Zero-denominator guard
# ===========================================================================

def test_player_with_zero_ab_has_none_avg():
    """If ab=0 for all games in window, w_avg must be None (not ZeroDivisionError)."""
    row = _stat_row(100, TODAY, ab=0, hits=0)
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)

    assert result is not None
    assert result.w_avg is None
    assert result.w_slg is None


# ===========================================================================
# Window boundary
# ===========================================================================

def test_game_outside_window_excluded():
    """A game exactly window_days back is excluded (window is exclusive at far end)."""
    # window_days=7 means days_back in [0, 6] are included; days_back=7 is excluded
    in_window = _stat_row(100, TODAY - timedelta(days=6), ab=4, hits=2)
    out_of_window = _stat_row(100, TODAY - timedelta(days=7), ab=10, hits=9)

    result = compute_rolling_window([in_window, out_of_window], as_of_date=TODAY, window_days=7)

    assert result is not None
    assert result.games_in_window == 1
    # Only the in-window row contributes
    assert result.w_ab == pytest.approx(0.95 ** 6 * 4)


def test_no_games_in_window_returns_none():
    """No rows within the window -> returns None."""
    old_row = _stat_row(100, TODAY - timedelta(days=31), ab=4, hits=2)
    result = compute_rolling_window([old_row], as_of_date=TODAY, window_days=7)

    assert result is None


# ===========================================================================
# Two-way player (Ohtani case)
# ===========================================================================

def test_ohtani_case_both_sets_populated():
    """Two-way player with both batting AB and innings pitched in same game."""
    row = _stat_row(
        bdl_player_id=999,
        game_date=TODAY,
        # Batting
        ab=3, hits=1, doubles=0, triples=0, home_runs=1,
        rbi=2, walks=0, strikeouts_bat=1, stolen_bases=0,
        # Pitching
        innings_pitched="5.0",
        earned_runs=1, hits_allowed=4, walks_allowed=1, strikeouts_pit=6,
    )
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)

    assert result is not None

    # Both batting and pitching fields should be populated
    assert result.w_ab is not None
    assert result.w_avg is not None
    assert result.w_ip is not None
    assert result.w_era is not None
    assert result.w_whip is not None

    # Spot-check values
    assert result.w_ab == pytest.approx(3.0)
    assert result.w_ip == pytest.approx(5.0)
    assert result.w_era == pytest.approx(9 * 1 / 5)


# ===========================================================================
# Batch computation
# ===========================================================================

def test_compute_all_returns_one_per_player_per_window():
    """2 players x 3 window sizes = 6 results (assuming each player has games in every window)."""
    p1_rows = [
        _stat_row(1, TODAY - timedelta(days=i), ab=3, hits=1)
        for i in range(5)
    ]
    p2_rows = [
        _stat_row(2, TODAY - timedelta(days=i), ab=4, hits=2)
        for i in range(5)
    ]

    results = compute_all_rolling_windows(
        p1_rows + p2_rows,
        as_of_date=TODAY,
        window_sizes=[7, 14, 30],
    )

    # Both players have 5 recent games -> appear in all three windows
    assert len(results) == 6
    player_ids = {r.bdl_player_id for r in results}
    window_sizes = {r.window_days for r in results}
    assert player_ids == {1, 2}
    assert window_sizes == {7, 14, 30}


def test_compute_all_skips_none_results():
    """Players with no games in a window are excluded from the flat results list."""
    # Player 1 has games in the past 5 days -> appears in 7d, 14d, 30d windows
    # Player 2 has only 1 game from 25 days ago -> only in 30d window, not 7d or 14d
    p1_rows = [
        _stat_row(1, TODAY - timedelta(days=i), ab=3, hits=1)
        for i in range(5)
    ]
    p2_rows = [
        _stat_row(2, TODAY - timedelta(days=25), ab=4, hits=2),
    ]

    results = compute_all_rolling_windows(
        p1_rows + p2_rows,
        as_of_date=TODAY,
        window_sizes=[7, 14, 30],
    )

    p1_results = [r for r in results if r.bdl_player_id == 1]
    p2_results = [r for r in results if r.bdl_player_id == 2]

    assert len(p1_results) == 3   # 7d, 14d, 30d
    assert len(p2_results) == 1   # 30d only (25 days back is in 30d but not 7d or 14d)
    assert p2_results[0].window_days == 30


def test_compute_all_default_window_sizes():
    """compute_all_rolling_windows defaults to [7, 14, 30] when window_sizes is None."""
    rows = [_stat_row(1, TODAY, ab=3, hits=1)]
    results = compute_all_rolling_windows(rows, as_of_date=TODAY)
    window_sizes = {r.window_days for r in results}
    assert window_sizes == {7, 14, 30}


def test_compute_all_empty_input():
    """Empty input returns empty list."""
    results = compute_all_rolling_windows([], as_of_date=TODAY, window_sizes=[7, 14, 30])
    assert results == []


# ===========================================================================
# IP edge cases
# ===========================================================================

def test_ip_parser_comprehensive():
    """Full suite of IP edge cases to verify correctness."""
    assert parse_ip(None) is None
    assert parse_ip("0") == 0.0
    assert parse_ip("0.0") == 0.0
    assert parse_ip("0.1") == pytest.approx(1 / 3)
    assert parse_ip("0.2") == pytest.approx(2 / 3)
    assert parse_ip("1.1") == pytest.approx(1 + 1 / 3)
    assert parse_ip("6.2") == pytest.approx(6 + 2 / 3)
    assert parse_ip("9.0") == pytest.approx(9.0)
    assert parse_ip("9") == 9.0


def test_parse_ip_six_two_is_not_six_point_two():
    """Critical correctness gate: 6.2 must NOT equal 6.2 decimal."""
    result = parse_ip("6.2")
    assert result != pytest.approx(6.2)
    assert result == pytest.approx(6 + 2 / 3, rel=1e-6)


# ===========================================================================
# V31: Runs (R) category
# ===========================================================================

def test_w_runs_accumulates_runs_scored():
    """w_runs should accumulate decay-weighted runs scored."""
    row = _stat_row(100, TODAY, ab=4, hits=2, runs=1)
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)

    assert result.w_runs == pytest.approx(1.0)


def test_w_runs_decay_applied():
    """w_runs should apply decay weighting to older games."""
    today_row = _stat_row(100, TODAY, ab=4, hits=2, runs=2)
    old_row = _stat_row(100, TODAY - timedelta(days=3), ab=3, hits=1, runs=1)

    result = compute_rolling_window([today_row, old_row], as_of_date=TODAY, window_days=7)

    w_today = 0.95 ** 0   # = 1.0
    w_old = 0.95 ** 3     # ~= 0.857375
    expected = w_today * 2 + w_old * 1
    assert result.w_runs == pytest.approx(expected, rel=1e-6)


# ===========================================================================
# V31: Total Bases (TB) category
# ===========================================================================

def test_w_tb_single():
    """Single counts as 1 TB."""
    row = _stat_row(100, TODAY, ab=1, hits=1, doubles=0, triples=0, home_runs=0)
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)
    assert result.w_tb == pytest.approx(1.0)


def test_w_tb_double():
    """Double counts as 2 TB."""
    row = _stat_row(100, TODAY, ab=1, hits=1, doubles=1, triples=0, home_runs=0)
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)
    assert result.w_tb == pytest.approx(2.0)


def test_w_tb_triple():
    """Triple counts as 3 TB."""
    row = _stat_row(100, TODAY, ab=1, hits=1, doubles=0, triples=1, home_runs=0)
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)
    assert result.w_tb == pytest.approx(3.0)


def test_w_tb_home_run():
    """Home run counts as 4 TB."""
    row = _stat_row(100, TODAY, ab=1, hits=1, doubles=0, triples=0, home_runs=1)
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)
    assert result.w_tb == pytest.approx(4.0)


def test_w_tb_mixed_hits():
    """TB = singles + 2*doubles + 3*triples + 4*HR."""
    # 3 singles = 3 TB, 1 double = 2 TB, 1 triple = 3 TB, 1 HR = 4 TB -> total 12 TB
    row = _stat_row(
        100, TODAY,
        ab=6, hits=6, doubles=1, triples=1, home_runs=1,
    )
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)
    # singles = hits - doubles - triples - HR = 6 - 1 - 1 - 1 = 3
    expected_tb = 3 * 1 + 1 * 2 + 1 * 3 + 1 * 4  # = 12
    assert result.w_tb == pytest.approx(expected_tb)


def test_w_tb_decay_applied():
    """w_tb should apply decay weighting to older games."""
    today_row = _stat_row(100, TODAY, ab=4, hits=2, home_runs=1)  # 1 singles + 1 HR = 5 TB
    old_row = _stat_row(100, TODAY - timedelta(days=5), ab=4, hits=1, doubles=1)  # 1 double = 2 TB

    result = compute_rolling_window([today_row, old_row], as_of_date=TODAY, window_days=7)

    w_today = 0.95 ** 0
    w_old = 0.95 ** 5
    expected = w_today * 5 + w_old * 2
    assert result.w_tb == pytest.approx(expected, rel=1e-6)


# ===========================================================================
# V31: Quality Starts (QS) category
# ===========================================================================

def test_w_qs_quality_start():
    """Quality start: IP >= 6 AND ER <= 3."""
    row = _stat_row(
        200, TODAY,
        innings_pitched="6.0", earned_runs=2,  # Meets QS criteria
        hits_allowed=5, walks_allowed=1, strikeouts_pit=5,
    )
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)
    assert result.w_qs == pytest.approx(1.0)


def test_w_qs_ip_too_low():
    """IP < 6 does NOT qualify as a QS regardless of ER."""
    row = _stat_row(
        200, TODAY,
        innings_pitched="5.2", earned_runs=0,  # Great game but only 5.2 IP
        hits_allowed=2, walks_allowed=0, strikeouts_pit=8,
    )
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)
    assert result.w_qs == pytest.approx(0.0)


def test_w_qs_too_many_runs():
    """ER > 3 does NOT qualify as a QS regardless of IP."""
    row = _stat_row(
        200, TODAY,
        innings_pitched="7.0", earned_runs=4,  # 7 IP but 4 ER
        hits_allowed=6, walks_allowed=2, strikeouts_pit=5,
    )
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)
    assert result.w_qs == pytest.approx(0.0)


def test_w_qs_exactly_6_ip_3_er():
    """Boundary case: IP=6.0 AND ER=3 IS a quality start."""
    row = _stat_row(
        200, TODAY,
        innings_pitched="6.0", earned_runs=3,  # Exactly meets criteria
        hits_allowed=5, walks_allowed=1, strikeouts_pit=4,
    )
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)
    assert result.w_qs == pytest.approx(1.0)


def test_w_qs_exactly_6_ip_4_er():
    """Boundary case: IP=6.0 AND ER=4 is NOT a quality start."""
    row = _stat_row(
        200, TODAY,
        innings_pitched="6.0", earned_runs=4,  # Fails by 1 ER
        hits_allowed=6, walks_allowed=2, strikeouts_pit=4,
    )
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)
    assert result.w_qs == pytest.approx(0.0)


def test_w_qs_ip_5_point_2():
    """Boundary case: IP=5.2 (5.667) is less than 6, not a QS."""
    row = _stat_row(
        200, TODAY,
        innings_pitched="5.2", earned_runs=1,  # Good ERA but short outing
        hits_allowed=4, walks_allowed=1, strikeouts_pit=6,
    )
    result = compute_rolling_window([row], as_of_date=TODAY, window_days=7)
    assert result.w_qs == pytest.approx(0.0)


def test_w_qs_decay_applied():
    """w_qs should apply decay weighting to older games."""
    qs_row = _stat_row(
        200, TODAY,
        innings_pitched="7.0", earned_runs=2,
    )
    non_qs_row = _stat_row(
        200, TODAY - timedelta(days=2),
        innings_pitched="6.0", earned_runs=4,  # Not a QS
    )

    result = compute_rolling_window([qs_row, non_qs_row], as_of_date=TODAY, window_days=7)

    w_today = 0.95 ** 0
    w_old = 0.95 ** 2
    expected = w_today * 1 + w_old * 0  # Only today's game is a QS
    assert result.w_qs == pytest.approx(expected, rel=1e-6)


def test_w_qs_multiple_games():
    """Multiple quality starts should accumulate with decay."""
    games = [
        _stat_row(200, TODAY - timedelta(days=i), innings_pitched="7.0", earned_runs=2)
        for i in range(5)  # 5 QS in a row
    ]

    result = compute_rolling_window(games, as_of_date=TODAY, window_days=7)

    # Sum of decay weights: 0.95^0 + 0.95^1 + 0.95^2 + 0.95^3 + 0.95^4
    expected = sum(0.95 ** i for i in range(5))
    assert result.w_qs == pytest.approx(expected, rel=1e-6)
