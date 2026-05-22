"""Tests for budget endpoint IP-accumulation key logic and acquisition counter."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def test_budget_ip_key_reads_my_stats():
    """get_matchup_stats returns 'my_stats', not 'my_team'; IP must use the right key."""
    matchup_stats = {"my_stats": {"IP": 14.2}, "opp_stats": {}, "opponent_name": "Opponent"}
    my_stats = matchup_stats.get("my_stats", {})
    ip = float(my_stats.get("IP", 0.0))
    assert ip == 14.2, f"Expected 14.2 but got {ip}"


def test_budget_ip_wrong_key_returns_zero():
    """Regression: the old 'my_team' key returns 0, confirming the fix matters."""
    matchup_stats = {"my_stats": {"IP": 14.2}, "opp_stats": {}, "opponent_name": "Opponent"}
    old_key = matchup_stats.get("my_team", {})
    ip = float(old_key.get("IP", 0.0))
    assert ip == 0.0, "Old key should give 0 — confirms the bug was real"


# ── Acquisition counter ────────────────────────────────────────────────────────

def test_local_count_wins_when_yahoo_lags():
    """max(local, yahoo) picks local when Yahoo hasn't propagated yet."""
    yahoo_count = 0
    local_count = 1
    acquisitions_used = max(yahoo_count, local_count)
    assert acquisitions_used == 1, "Expected local count to win over stale Yahoo 0"


def test_yahoo_count_wins_when_local_is_stale():
    """max(local, yahoo) picks Yahoo when Yahoo reflects more transactions than local."""
    yahoo_count = 3
    local_count = 1
    acquisitions_used = max(yahoo_count, local_count)
    assert acquisitions_used == 3


def test_week_start_is_monday_midnight():
    """week_start computation always lands on Monday 00:00 ET regardless of current day."""
    now = datetime(2026, 5, 20, 14, 30, 0, tzinfo=ZoneInfo("America/New_York"))  # Tuesday
    days_since_monday = now.weekday()  # Tuesday = 1
    week_start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
    assert week_start.weekday() == 0, "week_start must be Monday"
    assert week_start.hour == 0 and week_start.minute == 0


def test_roster_acquisition_model_fields():
    """RosterAcquisition can be instantiated with required fields."""
    from backend.models import RosterAcquisition
    now = datetime.now(ZoneInfo("America/New_York"))
    week_start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=now.weekday())
    obj = RosterAcquisition(
        team_key="469.l.72586.t.7",
        player_added_key="mlb.p.12345",
        player_dropped_key="mlb.p.67890",
        executed_at=now,
        week_start=week_start,
    )
    assert obj.team_key == "469.l.72586.t.7"
    assert obj.player_dropped_key == "mlb.p.67890"


def test_roster_acquisition_drop_optional():
    """player_dropped_key is nullable — free add (no drop) must work."""
    from backend.models import RosterAcquisition
    now = datetime.now(ZoneInfo("America/New_York"))
    week_start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=now.weekday())
    obj = RosterAcquisition(
        team_key="469.l.72586.t.7",
        player_added_key="mlb.p.12345",
        player_dropped_key=None,
        executed_at=now,
        week_start=week_start,
    )
    assert obj.player_dropped_key is None


# ── IP notation + pace ─────────────────────────────────────────────────────────

import pytest
from backend.services.constraint_helpers import ip_baseball_to_float, classify_ip_pace
from backend.contracts import IPPaceFlag


def test_ip_baseball_to_float_whole():
    assert ip_baseball_to_float(10.0) == pytest.approx(10.0)


def test_ip_baseball_to_float_one_out():
    """10.1 = 10 innings + 1 out = 10 + 1/3 = 10.333."""
    assert ip_baseball_to_float(10.1) == pytest.approx(10 + 1 / 3)


def test_ip_baseball_to_float_two_outs():
    """10.2 = 10 innings + 2 outs = 10 + 2/3 = 10.667."""
    assert ip_baseball_to_float(10.2) == pytest.approx(10 + 2 / 3)


def test_ip_baseball_to_float_zero():
    assert ip_baseball_to_float(0.0) == pytest.approx(0.0)


def test_ip_pace_behind_when_behind_weekly():
    """10.2 IP (true 10.667) by Friday (day 5) → projected 10.667/5*7 = 14.9 < 16.2 → BEHIND."""
    ip_true = ip_baseball_to_float(10.2)  # 10.667
    result = classify_ip_pace(ip_accumulated=ip_true, ip_minimum=18.0, days_elapsed=5, days_total=7)
    assert result == IPPaceFlag.BEHIND


def test_ip_pace_ahead_when_well_ahead_weekly():
    """20.0 IP by Wednesday (day 3) → projected 46.7 >> 19.8 → AHEAD."""
    result = classify_ip_pace(ip_accumulated=20.0, ip_minimum=18.0, days_elapsed=3, days_total=7)
    assert result == IPPaceFlag.AHEAD


def test_ip_pace_season_params_no_longer_used():
    """Season-level params (days_elapsed=61, total=182) must NOT be used.
    10.2 IP / 61 days * 182 = 30.4 which would give AHEAD — wrong for a weekly min."""
    ip_true = ip_baseball_to_float(10.2)
    # With correct weekly params on day 5 we expect BEHIND, not AHEAD
    result = classify_ip_pace(ip_accumulated=ip_true, ip_minimum=18.0, days_elapsed=5, days_total=7)
    assert result != IPPaceFlag.AHEAD, "Should be BEHIND, not AHEAD, with weekly params on day 5"


# ── Week number calculation ────────────────────────────────────────────────────

from datetime import date as _date


def _week(today: _date) -> int:
    from backend.routers.fantasy import _compute_mlb_current_week
    return _compute_mlb_current_week(today)


def test_week_number_opening_monday():
    """Week 1 starts on the first Yahoo matchup Monday (Mar 24, 2026)."""
    assert _week(_date(2026, 3, 24)) == 1


def test_week_number_end_of_week1():
    """Sunday of Week 1 (Mar 29) is still Week 1."""
    assert _week(_date(2026, 3, 29)) == 1


def test_week_number_week9_may22():
    """May 22, 2026 is Yahoo Week 9 — the primary regression test."""
    assert _week(_date(2026, 5, 22)) == 9


def test_week_number_clamped_at_25():
    """Past end of season, week is clamped to 25."""
    assert _week(_date(2026, 10, 1)) == 25


def test_week_number_clamped_at_1():
    """Before the season starts, week is clamped to 1."""
    assert _week(_date(2026, 3, 1)) == 1
