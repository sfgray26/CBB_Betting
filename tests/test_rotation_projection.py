"""Unit tests for the pure rotation-projection core (no DB).

Covers the 2-start surfacing mechanism, the +/-2 tolerance, official-probable
reconciliation, and median-cadence estimation. Spec:
reports/2026-07-22-streaming-rotation-projection-spec.md
"""
from datetime import date, timedelta

from backend.services.rotation_projection import (
    PitcherState,
    median_cadence,
    project_team_window,
)

D0 = date(2026, 7, 1)


def _day(n: int) -> date:
    return D0 + timedelta(days=n)


def _pitcher(name, last_offset, cadence=5, mlbam=None, ip=6.0):
    return PitcherState(
        team="NYY",
        bdl_player_id=None,
        mlbam_id=mlbam,
        pitcher_name=name,
        start_dates=[_day(last_offset)],
        typical_ip=ip,
        cadence=cadence,
    )


class TestMedianCadence:
    def test_strict_five_man(self):
        assert median_cadence([_day(0), _day(5), _day(10)]) == 5

    def test_six_day_turn(self):
        assert median_cadence([_day(0), _day(6), _day(12), _day(18)]) == 6

    def test_default_when_too_few_gaps(self):
        assert median_cadence([_day(0)]) == 6
        assert median_cadence([_day(0), _day(5)]) == 6  # only 1 gap

    def test_clamped_high(self):
        # gaps of 10 -> clamp to 8
        assert median_cadence([_day(0), _day(10), _day(20)]) == 8

    def test_clamped_low(self):
        # gaps of 3 -> clamp to 4
        assert median_cadence([_day(0), _day(3), _day(6)]) == 4

    def test_uses_last_five_starts_only(self):
        # Old wide gaps ignored; recent cadence is 6
        dates = [_day(0), _day(20), _day(40), _day(46), _day(52), _day(58)]
        assert median_cadence(dates) == 6


class TestProjectTwoStartSurfacing:
    def test_surfaces_two_start_pitchers(self):
        """A staggered 5-man rotation over an 8-day window produces 2-start arms.

        This is the core mechanism: advancing next_start after each assignment
        lets a pitcher be assigned again later in the window.
        """
        pitchers = [
            _pitcher("P1", -4, mlbam=1),
            _pitcher("P2", -3, mlbam=2),
            _pitcher("P3", -2, mlbam=3),
            _pitcher("P4", -1, mlbam=4),
            _pitcher("P5", -5, mlbam=5),
        ]
        game_dates = [_day(n) for n in range(8)]  # every day, 0..7

        assignments = project_team_window(pitchers, game_dates)

        # Every game date got a starter
        assert len(assignments) == 8
        assert {a.game_date for a in assignments} == set(game_dates)

        counts: dict[str, int] = {}
        for a in assignments:
            counts[a.pitcher.pitcher_name] = counts.get(a.pitcher.pitcher_name, 0) + 1
        two_start = [name for name, c in counts.items() if c >= 2]
        assert len(two_start) >= 1, counts
        # No pitcher assigned more than twice in an 8-day window (cadence >= 4)
        assert max(counts.values()) == 2


class TestTolerance:
    def test_rejects_outside_tolerance(self):
        p = _pitcher("Solo", -1, cadence=5)  # next_start = day 4
        # day 0 is 4 days from next_start -> outside +/-2
        assert project_team_window([p], [_day(0)]) == []

    def test_accepts_within_tolerance(self):
        p = _pitcher("Solo", -1, cadence=5)  # next_start = day 4
        out = project_team_window([p], [_day(2)])  # |4-2| = 2
        assert len(out) == 1
        assert out[0].pitcher.pitcher_name == "Solo"


class TestOfficialReconciliation:
    def test_official_date_not_projected_and_reanchors(self):
        p = _pitcher("Ace", -1, cadence=5, mlbam=99)  # next_start = day 4
        game_dates = [_day(0), _day(4), _day(5)]
        # Without official: day 4 is assignable
        assert any(a.game_date == _day(4) for a in project_team_window([p], game_dates))

        # With official on day 4 keyed to this pitcher: day 4 is skipped and the
        # pitcher re-anchors to day 4 + cadence = day 9, so day 5 (|9-5|=4) is not
        # projected either.
        p2 = _pitcher("Ace", -1, cadence=5, mlbam=99)
        out = project_team_window([p2], game_dates, official_dates={_day(4): 99})
        assert all(a.game_date != _day(4) for a in out)
        assert out == []

    def test_empty_rotation_is_safe(self):
        assert project_team_window([], [_day(0), _day(1)]) == []
