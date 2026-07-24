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
    """In-memory SQLite session with the tables the resolver reads. mlb_game_log
    is created via raw SQL (its JSONB column can't compile on SQLite; the resolver
    only reads game_id + home/away_team_id)."""
    from sqlalchemy import text
    engine = create_engine("sqlite:///:memory:")
    MLBPlayerStats.__table__.create(engine)
    PlayerIDMapping.__table__.create(engine)
    from backend.models import MLBTeam
    MLBTeam.__table__.create(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE mlb_game_log ("
            " game_id INTEGER PRIMARY KEY,"
            " home_team_id INTEGER,"
            " away_team_id INTEGER)"
        ))
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        yield db
    finally:
        db.close()


_row_id = [0]


def _team(db, team_id, abbr):
    from backend.models import MLBTeam
    db.add(MLBTeam(team_id=team_id, abbreviation=abbr, name=abbr, display_name=abbr,
                   short_name=abbr, location=abbr, slug=abbr.lower(),
                   league="National", division="West"))


def _game(db, game_id, home_id, away_id):
    from sqlalchemy import text
    db.execute(text(
        "INSERT INTO mlb_game_log (game_id, home_team_id, away_team_id) "
        "VALUES (:g, :h, :a)"
    ), {"g": game_id, "h": home_id, "a": away_id})


def _insert_start(db, bdl_id, name, game_id, game_date, ip="6.2"):
    """Insert an MLBPlayerStats row in the TRUE PRODUCTION shape: NO team anywhere
    in raw_payload (both player.team and top-level team null). Team is derivable
    only via game_id -> mlb_game_log."""
    _row_id[0] += 1
    db.add(MLBPlayerStats(
        id=_row_id[0],
        bdl_player_id=bdl_id,
        game_id=game_id,
        game_date=game_date,
        season=2026,
        innings_pitched=ip,
        era=3.0,
        raw_payload={
            "player": {"id": bdl_id, "full_name": name, "position": "P", "team": None},
            "team": None,
            "ip": ip,
        },
    ))


class TestBuildRotationSetsProductionShape:
    """Team is null everywhere in raw_payload (real prod); it must be derived from
    game membership (MLBPlayerStats.game_id -> mlb_game_log -> mlb_team)."""

    def _seed_teams(self, db):
        _team(db, 1, "LAD")
        _team(db, 2, "SF")
        _team(db, 3, "SD")
        _team(db, 4, "COL")

    def test_team_resolved_from_game_membership(self, rot_db):
        """REGRESSION (prod ground truth 2026-07-24): raw_payload carries no team;
        build_rotation_sets must resolve LAD from the pitcher's games. Before this
        fix build_rotation_sets returned {} -> projected_records:0, d2_d5_total:0."""
        today = date(2026, 7, 24)
        db = rot_db
        self._seed_teams(db)
        db.add(PlayerIDMapping(bdl_id=42, mlbam_id=999, full_name="Ace Pitcher",
                               normalized_name="ace pitcher", source="manual"))
        # LAD(1) starter facing 3 distinct opponents so the modal team is unique.
        _game(db, 100, 1, 2)   # LAD home vs SF
        _game(db, 101, 3, 1)   # LAD away at SD
        _game(db, 102, 1, 4)   # LAD home vs COL
        _insert_start(db, 42, "Ace Pitcher", 100, today - timedelta(days=12))
        _insert_start(db, 42, "Ace Pitcher", 101, today - timedelta(days=7))
        _insert_start(db, 42, "Ace Pitcher", 102, today - timedelta(days=2))
        db.commit()

        rotation = build_rotation_sets(db, today)
        assert "LAD" in rotation, rotation
        p = rotation["LAD"][0]
        assert p.pitcher_name == "Ace Pitcher"
        assert p.mlbam_id == 999
        assert p.cadence == 5

    def test_single_opponent_is_unresolved(self, rot_db):
        """A pitcher whose only games are vs one opponent is ambiguous (both teams
        tie) and is left unresolved rather than mis-assigned."""
        today = date(2026, 7, 24)
        db = rot_db
        self._seed_teams(db)
        _game(db, 200, 1, 2)   # LAD vs SF
        _game(db, 201, 2, 1)   # SF vs LAD  (same two teams)
        _insert_start(db, 55, "Ambiguous Arm", 200, today - timedelta(days=7))
        _insert_start(db, 55, "Ambiguous Arm", 201, today - timedelta(days=2))
        db.commit()
        rotation = build_rotation_sets(db, today)
        # LAD=2 and SF=2 tie -> unresolved -> pitcher excluded from every team
        assert all("Ambiguous Arm" not in [p.pitcher_name for p in v]
                   for v in rotation.values()), rotation

    def test_end_to_end_two_start_projection(self, rot_db):
        """build_rotation_sets -> project surfaces a 2-start pitcher over a window,
        with team derived purely from game membership."""
        today = date(2026, 7, 24)
        db = rot_db
        self._seed_teams(db)
        opponents = [2, 3, 4]
        gid = 300
        # 5 LAD starters, each 3 starts vs varied opponents ~5 days apart.
        rotation_arms = [(10, "P1", 4), (11, "P2", 3), (12, "P3", 2),
                         (13, "P4", 1), (14, "P5", 5)]
        for bdl, name, last_off in rotation_arms:
            db.add(PlayerIDMapping(bdl_id=bdl, mlbam_id=900 + bdl, full_name=name,
                                   normalized_name=name.lower(), source="manual"))
            for i, prior in enumerate((last_off + 10, last_off + 5, last_off)):
                gid += 1
                _game(db, gid, 1, opponents[i])  # LAD home vs a varied opponent
                _insert_start(db, bdl, name, gid, today - timedelta(days=prior))
        db.commit()

        game_dates = [today + timedelta(days=n) for n in range(8)]
        projected = project_probable_starters(db, today, {"LAD": game_dates})
        counts: dict = {}
        for (team, d), p in projected.items():
            counts[p.pitcher_name] = counts.get(p.pitcher_name, 0) + 1
        assert projected, "expected projected starters, got none"
        assert max(counts.values()) >= 2, counts

    def test_backtest_actuals_resolve_team(self, rot_db):
        """REGRESSION: backtest actual-start extraction must resolve team from game
        membership too, else d2_d5_total is always 0 (no evaluable sample)."""
        db = rot_db
        self._seed_teams(db)
        d = date(2026, 7, 20)
        _game(db, 400, 1, 2)   # LAD vs SF
        _game(db, 401, 3, 1)   # SD vs LAD (2nd distinct opponent -> resolvable)
        _insert_start(db, 42, "Ace Pitcher", 400, d)
        _insert_start(db, 42, "Ace Pitcher", 401, d - timedelta(days=5))
        db.commit()
        actuals = _actual_starts_by_team_date(db, d - timedelta(days=5), d)
        assert ("LAD", d) in actuals, actuals
        assert "ace pitcher" in actuals[("LAD", d)]

    def test_backtest_has_evaluable_sample(self, rot_db):
        """REGRESSION for the exact production symptom (d2_d5_total:0): with the
        game-membership team derivation the backtest finds a non-empty evaluable
        sample. Seeds one LAD starter with a steady 5-day cadence vs varied
        opponents over ~45 days."""
        from backend.services.rotation_projection import backtest_rotation_projection
        db = rot_db
        self._seed_teams(db)
        today = date(2026, 7, 24)
        opponents = [2, 3, 4]  # SF, SD, COL — varied so team resolves
        gid = 500
        db.add(PlayerIDMapping(bdl_id=42, mlbam_id=999, full_name="Ace Pitcher",
                               normalized_name="ace pitcher", source="manual"))
        for i, off in enumerate(range(0, 45, 5)):  # starts 0,5,10,...,40 days ago
            gid += 1
            _game(db, gid, 1, opponents[i % len(opponents)])
            _insert_start(db, 42, "Ace Pitcher", gid, today - timedelta(days=off))
        db.commit()

        result = backtest_rotation_projection(db, days=12, horizon=7)
        assert result["anchor_date"] == today.isoformat()
        assert result["d2_d5_total"] > 0, result  # was 0 in production
