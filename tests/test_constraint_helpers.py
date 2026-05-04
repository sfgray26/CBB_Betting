"""Tests for constraint helpers — 7 data gap closure functions."""

import pytest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from backend.services.constraint_helpers import (
    count_weekly_acquisitions,
    extract_ip_from_scoreboard,
    classify_ip_pace,
    count_games_remaining,
    extract_team_record,
    lookup_opposing_sp,
    OpposingSPInfo,
    resolve_playing_status,
)
from backend.contracts import IPPaceFlag


# =============================================================================
# count_weekly_acquisitions tests
# =============================================================================

def test_count_weekly_acquisitions_basic():
    """3 adds in window, 1 outside → returns 3."""
    transactions = [
        {"type": "add", "destination_team_key": "mlb.l.123.t.1", "timestamp": 20},
        {"type": "add", "destination_team_key": "mlb.l.123.t.1", "timestamp": 25},
        {"type": "add", "destination_team_key": "mlb.l.123.t.1", "timestamp": 30},
        {"type": "add", "destination_team_key": "mlb.l.123.t.1", "timestamp": 9999},  # outside
    ]
    week_start = datetime(1970, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    week_end = datetime(1970, 1, 1, 0, 0, 50, tzinfo=timezone.utc)

    result = count_weekly_acquisitions(transactions, "mlb.l.123.t.1", week_start, week_end)
    assert result == 3


def test_count_weekly_acquisitions_filters_other_teams():
    """Adds by other teams excluded."""
    transactions = [
        {"type": "add", "destination_team_key": "mlb.l.123.t.1", "timestamp": 20},
        {"type": "add", "destination_team_key": "mlb.l.123.t.2", "timestamp": 30},
        {"type": "add", "destination_team_key": "mlb.l.123.t.3", "timestamp": 40},
    ]
    week_start = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    week_end = datetime(1970, 1, 1, 0, 1, 0, tzinfo=timezone.utc)

    result = count_weekly_acquisitions(transactions, "mlb.l.123.t.1", week_start, week_end)
    assert result == 1


def test_count_weekly_acquisitions_filters_drops():
    """Drop transactions excluded."""
    transactions = [
        {"type": "add", "destination_team_key": "mlb.l.123.t.1", "timestamp": 20},
        {"type": "drop", "destination_team_key": "mlb.l.123.t.1", "timestamp": 30},
        {"type": "add", "destination_team_key": "mlb.l.123.t.1", "timestamp": 40},
    ]
    week_start = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    week_end = datetime(1970, 1, 1, 0, 1, 0, tzinfo=timezone.utc)

    result = count_weekly_acquisitions(transactions, "mlb.l.123.t.1", week_start, week_end)
    assert result == 2


def test_count_weekly_acquisitions_nested_destination():
    """Handles nested destination_team structure."""
    transactions = [
        {
            "type": "add",
            "destination_team": {"team_key": "mlb.l.123.t.1"},
            "timestamp": 30,
        },
    ]
    week_start = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    week_end = datetime(1970, 1, 1, 0, 1, 0, tzinfo=timezone.utc)

    result = count_weekly_acquisitions(transactions, "mlb.l.123.t.1", week_start, week_end)
    assert result == 1


# =============================================================================
# extract_ip_from_scoreboard tests
# =============================================================================

def test_extract_ip_from_scoreboard():
    """{"IP": 15.2, "ERA": 3.50} → returns 15.2."""
    matchup_stats = {"IP": 15.2, "ERA": 3.50, "W": 2}
    result = extract_ip_from_scoreboard(matchup_stats)
    assert result == 15.2


def test_extract_ip_missing():
    """Empty dict → returns 0.0."""
    result = extract_ip_from_scoreboard({})
    assert result == 0.0


# =============================================================================
# classify_ip_pace tests
# =============================================================================

def test_classify_ip_pace_behind():
    """5.0 IP after 4 days → BEHIND."""
    result = classify_ip_pace(5.0, 18.0, 4, 7)
    assert result == IPPaceFlag.BEHIND


def test_classify_ip_pace_on_track():
    """10.5 IP after 4 days → ON_TRACK."""
    # 10.5/4 * 7 = 18.375 projected, which is within 10% of 18 (16.2-19.8)
    result = classify_ip_pace(10.5, 18.0, 4, 7)
    assert result == IPPaceFlag.ON_TRACK


def test_classify_ip_pace_ahead():
    """16.0 IP after 4 days → AHEAD."""
    # 16/4 * 7 = 28 projected, which exceeds 19.8
    result = classify_ip_pace(16.0, 18.0, 4, 7)
    assert result == IPPaceFlag.AHEAD


def test_classify_ip_pace_day_zero():
    """0 days elapsed → BEHIND."""
    result = classify_ip_pace(0.0, 18.0, 0, 7)
    assert result == IPPaceFlag.BEHIND


# =============================================================================
# count_games_remaining tests
# =============================================================================

def test_count_games_remaining():
    """Team with 3 games left in week → returns 3."""
    today = datetime(2026, 4, 15, tzinfo=ZoneInfo("America/New_York"))
    week_end = datetime(2026, 4, 19, 23, 59, tzinfo=ZoneInfo("America/New_York"))
    schedule = {
        "NYY": [
            datetime(2026, 4, 15, 19, tzinfo=ZoneInfo("America/New_York")),
            datetime(2026, 4, 16, 19, tzinfo=ZoneInfo("America/New_York")),
            datetime(2026, 4, 18, 19, tzinfo=ZoneInfo("America/New_York")),
            datetime(2026, 4, 20, 19, tzinfo=ZoneInfo("America/New_York")),  # after week_end
        ]
    }

    result = count_games_remaining("NYY", schedule, today, week_end)
    assert result == 3


def test_count_games_remaining_no_games():
    """Team not in schedule → returns 0."""
    today = datetime(2026, 4, 15, tzinfo=ZoneInfo("America/New_York"))
    week_end = datetime(2026, 4, 19, 23, 59, tzinfo=ZoneInfo("America/New_York"))
    schedule = {}

    result = count_games_remaining("BOS", schedule, today, week_end)
    assert result == 0


# =============================================================================
# extract_team_record tests
# =============================================================================

def test_extract_team_record():
    """Standings with my team at 5-3-1 → returns (5, 3, 1)."""
    standings = [
        {
            "team": {
                "team_key": "mlb.l.123.t.1",
                "outcome_totals": {"wins": 5, "losses": 3, "ties": 1}
            }
        },
        {
            "team": {
                "team_key": "mlb.l.123.t.2",
                "outcome_totals": {"wins": 4, "losses": 4, "ties": 1}
            }
        },
    ]

    result = extract_team_record(standings, "mlb.l.123.t.1")
    assert result == (5, 3, 1)


def test_extract_team_record_not_found():
    """My team not in standings → returns (0, 0, 0)."""
    standings = [
        {
            "team": {
                "team_key": "mlb.l.123.t.2",
                "outcome_totals": {"wins": 4, "losses": 4, "ties": 1}
            }
        },
    ]

    result = extract_team_record(standings, "mlb.l.123.t.1")
    assert result == (0, 0, 0)


# =============================================================================
# lookup_opposing_sp tests
# =============================================================================

def test_lookup_opposing_sp_found():
    """Hitter vs known SP → returns OpposingSPInfo."""
    schedule_entry = {
        "home_team": "NYY",
        "away_team": "BOS",
    }
    probable_pitchers = [
        {"team": "BOS", "name": "Chris Sale", "handedness": "L"},
        {"team": "NYY", "name": "Gerrit Cole", "handedness": "R"},
    ]

    result = lookup_opposing_sp("NYY", schedule_entry, probable_pitchers)
    assert result is not None
    assert result.sp_name == "Chris Sale"
    assert result.sp_handedness == "L"
    assert result.opponent_team == "BOS"
    assert result.home_away == "home"


def test_lookup_opposing_sp_no_game():
    """No game today → returns None."""
    result = lookup_opposing_sp("NYY", None, [])
    assert result is None


def test_lookup_opposing_sp_no_probable_data():
    """Game found but no probable pitcher → returns OpposingSPInfo with None SP."""
    schedule_entry = {
        "home_team": "NYY",
        "away_team": "BOS",
    }

    result = lookup_opposing_sp("NYY", schedule_entry, [])
    assert result is not None
    assert result.sp_name is None
    assert result.sp_handedness is None
    assert result.opponent_team == "BOS"
    assert result.home_away == "home"


# =============================================================================
# resolve_playing_status tests
# =============================================================================

def test_resolve_playing_status_il():
    """IL status → "IL"."""
    roster_entry = {"status": "IL"}
    result = resolve_playing_status(roster_entry, True)
    assert result == "IL"


def test_resolve_playing_status_il10():
    """IL10 status → "IL"."""
    roster_entry = {"status": "IL10"}
    result = resolve_playing_status(roster_entry, True)
    assert result == "IL"


def test_resolve_playing_status_minors():
    """Minors status → "minors"."""
    roster_entry = {"status": "minors"}
    result = resolve_playing_status(roster_entry, True)
    assert result == "minors"


def test_resolve_playing_status_na():
    """NA status → "minors"."""
    roster_entry = {"status": "NA"}
    result = resolve_playing_status(roster_entry, True)
    assert result == "minors"


def test_resolve_playing_status_no_game():
    """No game today → "not_playing"."""
    roster_entry = {"status": "active"}
    result = resolve_playing_status(roster_entry, False)
    assert result == "not_playing"


def test_resolve_playing_status_playing():
    """Has game, not IL → "playing"."""
    roster_entry = {"status": "active"}
    result = resolve_playing_status(roster_entry, True)
    assert result == "playing"
