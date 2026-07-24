"""Unit tests for the pure rotation-projection core (no DB) plus DB-backed
regression tests for production starter-history extraction.

Covers the 2-start surfacing mechanism, the +/-2 tolerance, official-probable
reconciliation, median-cadence estimation, and — critically — the production bug
where `MLBPlayerStats.raw_payload` nests the team under `player.team` (top-level
`team` is null), which had left rotation sets and the backtest empty. Spec:
reports/2026-07-22-streaming-rotation-projection-spec.md
"""
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models import MLBPlayerStats, PlayerIDMapping
from backend.services.rotation_projection import (
    PitcherState,
    build_rotation_sets,
    median_cadence,
    project_team_window,
    project_probable_starters,
    _actual_starts_by_team_date,
    _starter_team_name,
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


# ---------------------------------------------------------------------------
# Production-shape extraction (regression for zero projected_records / zero
# backtest sample — team is nested under player.team, top-level team is null)
# ---------------------------------------------------------------------------

class TestStarterTeamNameExtraction:
    def test_reads_nested_player_team(self):
        """BDL /stats nests team under player; top-level team is null (prod shape)."""
        payload = {
            "player": {
                "full_name": "Ace Pitcher",
                "team": {"abbreviation": "LAD"},
            },
            "team": None,
        }
        assert _starter_team_name(payload) == ("LAD", "Ace Pitcher")

    def test_falls_back_to_top_level_team(self):
        payload = {
            "player": {"full_name": "Ace Pitcher"},
            "team": {"abbreviation": "NYY"},
        }
        assert _starter_team_name(payload) == ("NYY", "Ace Pitcher")

    def test_normalizes_alias(self):
        payload = {"player": {"full_name": "X", "team": {"abbreviation": "TBR"}}}
        assert _starter_team_name(payload) == ("TB", "X")

    def test_missing_everything_is_empty(self):
        assert _starter_team_name({}) == ("", "")
        assert _starter_team_name(None) == ("", "")


@pytest.fixture
def rot_db():
    """In-memory SQLite session with just the tables build_rotation_sets reads."""
    engine = create_engine("sqlite:///:memory:")
    MLBPlayerStats.__table__.create(engine)
    PlayerIDMapping.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        yield db
    finally:
        db.close()


_row_id = [0]


def _insert_start(db, bdl_id, name, team, game_date, ip="6.2"):
    """Insert an MLBPlayerStats row with the PRODUCTION raw_payload shape:
    team nested under player.team, top-level team null."""
    _row_id[0] += 1
    db.add(MLBPlayerStats(
        id=_row_id[0],
        bdl_player_id=bdl_id,
        game_id=None,
        game_date=game_date,
        season=2026,
        innings_pitched=ip,
        era=3.0,
        raw_payload={
            "player": {"id": bdl_id, "full_name": name, "position": "P",
                       "team": {"abbreviation": team}},
            "team": None,
            "ip": ip,
        },
    ))


class TestBuildRotationSetsProductionShape:
    def test_rotation_set_built_from_nested_team(self, rot_db):
        """REGRESSION: build_rotation_sets must find pitchers when team is nested
        under player.team (prod). Before the fix this returned {} → zero projected
        rows and zero backtest sample."""
        today = date(2026, 7, 24)
        db = rot_db
        db.add(PlayerIDMapping(bdl_id=42, mlbam_id=999, full_name="Ace Pitcher",
                               normalized_name="ace pitcher", source="manual"))
        # A 5-man-ish cadence: starts 12, 7, 2 days ago
        for off in (12, 7, 2):
            _insert_start(db, 42, "Ace Pitcher", "LAD", today - timedelta(days=off))
        db.commit()

        rotation = build_rotation_sets(db, today)
        assert "LAD" in rotation, rotation
        pitchers = rotation["LAD"]
        assert len(pitchers) == 1
        p = pitchers[0]
        assert p.pitcher_name == "Ace Pitcher"
        assert p.mlbam_id == 999
        assert p.cadence == 5  # median gap of [5, 5]

    def test_end_to_end_two_start_projection(self, rot_db):
        """build_rotation_sets → project surfaces a 2-start pitcher over a window."""
        today = date(2026, 7, 24)
        db = rot_db
        # Full 5-man rotation, each with a recent start ~5 days apart
        rotation_arms = [
            (10, "P1", 4), (11, "P2", 3), (12, "P3", 2), (13, "P4", 1), (14, "P5", 5),
        ]
        for bdl, name, last_off in rotation_arms:
            db.add(PlayerIDMapping(bdl_id=bdl, mlbam_id=900 + bdl, full_name=name,
                                   normalized_name=name.lower(), source="manual"))
            for prior in (last_off + 10, last_off + 5, last_off):
                _insert_start(db, bdl, name, "LAD", today - timedelta(days=prior))
        db.commit()

        # Team plays every day for 8 days
        game_dates = [today + timedelta(days=n) for n in range(8)]
        projected = project_probable_starters(db, today, {"LAD": game_dates})
        # Some pitcher must get two projected starts in the window
        counts: dict = {}
        for (team, d), p in projected.items():
            counts[p.pitcher_name] = counts.get(p.pitcher_name, 0) + 1
        assert projected, "expected projected starters, got none"
        assert max(counts.values()) >= 2, counts

    def test_backtest_actuals_read_nested_team(self, rot_db):
        """REGRESSION: backtest actual-start extraction must read nested team too,
        else d2_d5_total is always 0 (no evaluable sample)."""
        db = rot_db
        d = date(2026, 7, 20)
        _insert_start(db, 42, "Ace Pitcher", "LAD", d)
        db.commit()
        actuals = _actual_starts_by_team_date(db, d, d)
        assert ("LAD", d) in actuals
        assert "ace pitcher" in actuals[("LAD", d)]
