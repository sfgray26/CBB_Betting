"""
Integration tests for cat_scores backfill logic using SQLite in-memory.

Tests the backend.services.cat_scores_builder module with a real database
session, avoiding the fragile ORM mocking anti-pattern that caused prior
tests to hang.
"""

import json
from datetime import datetime

import pytest
from sqlalchemy import create_engine, TypeDecorator, JSON, Column
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSONB

from backend.models import Base, PlayerProjection
from backend.services.cat_scores_builder import (
    BATTER_POS,
    BATTER_WEIGHTS,
    PITCHER_POS,
    PITCHER_WEIGHTS,
    classify_player,
    compute_cat_scores,
    run_backfill,
)


# ---------------------------------------------------------------------------
# SQLite compatibility layer for JSONB
# ---------------------------------------------------------------------------

class JSONBForSQLite(TypeDecorator):
    """TypeDecorator that uses JSONB for PostgreSQL, JSON for SQLite."""
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        return json.dumps(value) if value is not None else None

    def process_result_value(self, value, dialect):
        return json.loads(value) if value is not None else None


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database with player_projections table."""
    engine = create_engine("sqlite:///:memory:")

    # Patch ALL JSONB columns to JSON for SQLite compatibility
    # Save original types for restoration
    original_types = {}
    for table in Base.metadata.sorted_tables:
        for col in table.columns:
            if str(col.type) == 'JSONB':
                original_types[(table.name, col.name)] = col.type
                col.type = JSON()

    try:
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        try:
            yield db
        finally:
            db.close()
    finally:
        # Restore original types
        for (table_name, col_name), orig_type in original_types.items():
            table = Base.metadata.tables[table_name]
            table.columns[col_name].type = orig_type


def test_classify_batter():
    assert classify_player(["OF"], None, 25, 80) == "batter"
    assert classify_player(["1B", "3B"], None, 30, 90) == "batter"
    assert classify_player(["SS"], 4.00, 35, 100) == "batter"


def test_classify_pitcher():
    assert classify_player(["SP"], 3.25, 5, 5) == "pitcher"
    assert classify_player(["RP"], 2.80, 0, 0) == "pitcher"
    assert classify_player([], 3.50, 2, 5) == "pitcher"  # ERA heuristic


def test_classify_ambiguous():
    assert classify_player([], 4.00, 0, 0) is None
    assert classify_player([], None, 2, 5) is None


def test_classify_mixed_position():
    # Two-way player with non-default ERA → pitcher
    assert classify_player(["P", "1B"], 3.50, 10, 30) == "pitcher"
    # Two-way player with default ERA → batter (default to batting value)
    assert classify_player(["P", "OF"], 4.00, 20, 60) == "batter"


def test_compute_cat_scores_batters(in_memory_db):
    """Test z-score computation for batters."""
    # Create test players
    p1 = PlayerProjection(
        player_id="p1", player_name="Batter A", team="NYY", positions=["OF"],
        hr=30, r=90, rbi=85, sb=10, avg=0.280, slg=0.500, ops=0.850,
        era=None, whip=None, k_per_nine=None, cat_scores={}
    )
    p2 = PlayerProjection(
        player_id="p2", player_name="Batter B", team="BOS", positions=["SS"],
        hr=15, r=60, rbi=55, sb=5, avg=0.250, slg=0.400, ops=0.720,
        era=None, whip=None, k_per_nine=None, cat_scores={}
    )
    in_memory_db.add(p1)
    in_memory_db.add(p2)
    in_memory_db.commit()

    # Build player dicts for scoring
    players = [
        {
            "player_id": "p1",
            "proj": {
                "r": 90, "h": 150, "hr": 30, "rbi": 85, "k_bat": 100,
                "tb": 250, "avg": 0.280, "ops": 0.850, "nsb": 10,
            },
            "cat_scores": {},
        },
        {
            "player_id": "p2",
            "proj": {
                "r": 60, "h": 120, "hr": 15, "rbi": 55, "k_bat": 80,
                "tb": 200, "avg": 0.250, "ops": 0.720, "nsb": 5,
            },
            "cat_scores": {},
        },
    ]

    compute_cat_scores(players, BATTER_WEIGHTS)

    # Both players should have cat_scores with all 9 categories
    for p in players:
        assert len(p["cat_scores"]) == 9
        assert all(isinstance(v, float) for v in p["cat_scores"].values())
        assert "z_score" in p

    # Player 1 (better stats) should have higher z_score
    assert players[0]["z_score"] > players[1]["z_score"]


def test_compute_cat_scores_pitchers(in_memory_db):
    """Test z-score computation for pitchers."""
    players = [
        {
            "player_id": "sp1",
            "proj": {
                "w": 15, "l": 5, "hr_pit": 20, "k_pit": 200,
                "era": 3.00, "whip": 1.10, "k9": 10.0, "qs": 20, "nsv": 5,
            },
            "cat_scores": {},
        },
        {
            "player_id": "sp2",
            "proj": {
                "w": 10, "l": 10, "hr_pit": 30, "k_pit": 150,
                "era": 4.50, "whip": 1.40, "k9": 8.0, "qs": 12, "nsv": 2,
            },
            "cat_scores": {},
        },
    ]

    compute_cat_scores(players, PITCHER_WEIGHTS)

    for p in players:
        assert len(p["cat_scores"]) == 9
        assert "z_score" in p

    # Player 1 (better ERA/WHIP, more W) should have higher z_score
    assert players[0]["z_score"] > players[1]["z_score"]


def test_run_backfill_full_pipeline(in_memory_db):
    """End-to-end test of the backfill pipeline with SQLite in-memory.

    Covers the five fixture cases from the Execution Blueprint:
    - Row A: batter with full stats, empty cat_scores → should be populated
    - Row B: pitcher with full stats, empty cat_scores → should be populated
    - Row C: batter with placeholder stats (hr=0, r=0), empty cat_scores → should be populated
    - Row D: ambiguous row (no positions, default ERA), empty cat_scores → should be populated with pitcher defaults
    - Row E: already populated (non-empty cat_scores) → should be skipped
    """
    # Row A: Full batter stats
    a = PlayerProjection(
        player_id="row_a", player_name="Batter Full", team="NYY", positions=["OF"],
        hr=25, r=80, rbi=75, sb=8, avg=0.270, slg=0.480, ops=0.810,
        era=None, whip=None, k_per_nine=None, cat_scores={}
    )
    # Row B: Full pitcher stats
    b = PlayerProjection(
        player_id="row_b", player_name="Pitcher Full", team="LAD", positions=["SP"],
        hr=0, r=0, rbi=0, sb=0, avg=None, slg=None, ops=None,
        era=3.25, whip=1.15, k_per_nine=9.5, cat_scores={}
    )
    # Row C: Batter with placeholder stats (hr=0, r=0) — must still get cat_scores
    c = PlayerProjection(
        player_id="row_c", player_name="Batter Placeholder", team="BOS", positions=["1B"],
        hr=0, r=0, rbi=0, sb=0, avg=0.250, slg=0.350, ops=0.650,
        era=None, whip=None, k_per_nine=None, cat_scores={}
    )
    # Row D: Ambiguous row — no positions, default ERA
    d = PlayerProjection(
        player_id="row_d", player_name="Ambiguous", team=None, positions=None,
        hr=0, r=0, rbi=0, sb=0, avg=None, slg=None, ops=None,
        era=4.00, whip=1.30, k_per_nine=8.0, cat_scores={}
    )
    # Row E: Already populated — should be SKIPPED
    e = PlayerProjection(
        player_id="row_e", player_name="Already Filled", team="CHC", positions=["C"],
        hr=15, r=50, rbi=40, sb=2, avg=0.240, slg=0.400, ops=0.700,
        era=None, whip=None, k_per_nine=None,
        cat_scores={"r": 0.5, "h": 0.3, "hr": 1.2, "rbi": 0.8, "k_bat": -0.4,
                    "tb": 0.6, "avg": 0.9, "ops": 1.0, "nsb": 0.2}
    )

    in_memory_db.add_all([a, b, c, d, e])
    in_memory_db.commit()

    # Run the backfill pipeline
    result = run_backfill(in_memory_db)

    # Verify result structure
    assert result["status"] == "success"
    assert result["cat_scores_updated"] == 4  # A, B, C, D (E is skipped)
    assert result["skipped_already_filled"] == 1  # E
    assert result["target_met"] is True
    assert result["verify_remaining_empty"] == 0

    # Verify database state
    in_memory_db.expire_all()  # Clear session cache
    rows = in_memory_db.query(PlayerProjection).order_by(PlayerProjection.player_id).all()
    assert len(rows) == 5

    row_by_id = {r.player_id: r for r in rows}

    # Row A: populated with 9-category batter scores
    assert len(row_by_id["row_a"].cat_scores or {}) == 9
    assert "z_score" not in row_by_id["row_a"].cat_scores  # z_score is separate in DB

    # Row B: populated with 9-category pitcher scores
    assert len(row_by_id["row_b"].cat_scores or {}) == 9

    # Row C: populated despite placeholder stats
    assert len(row_by_id["row_c"].cat_scores or {}) == 9

    # Row D: ambiguous gets pitcher defaults
    assert len(row_by_id["row_d"].cat_scores or {}) == 9

    # Row E: UNCHANGED (skipped)
    assert row_by_id["row_e"].cat_scores == {"r": 0.5, "h": 0.3, "hr": 1.2, "rbi": 0.8,
                                               "k_bat": -0.4, "tb": 0.6, "avg": 0.9,
                                               "ops": 1.0, "nsb": 0.2}


def test_run_backfill_team_lookup_from_statcast(in_memory_db):
    """Test that null team fields are resolved from statcast_performances."""
    from sqlalchemy import text
    from datetime import date

    # Create a player with null team and a corresponding statcast row
    p = PlayerProjection(
        player_id="teamless", player_name="Teamless", team=None, positions=["OF"],
        hr=20, r=70, rbi=60, sb=5, avg=0.260, slg=0.450, ops=0.780,
        era=None, whip=None, k_per_nine=None, cat_scores={}
    )
    in_memory_db.add(p)

    # Insert a statcast row with required columns (table already created by Base.metadata.create_all)
    in_memory_db.execute(
        text("INSERT INTO statcast_performances (player_id, player_name, team, game_date) "
             "VALUES ('teamless', 'Teamless', 'NYY', '2026-04-01')")
    )
    in_memory_db.commit()

    # Run backfill
    result = run_backfill(in_memory_db)

    assert result["status"] == "success"
    assert result["team_updated"] == 1

    # Verify team field was populated
    in_memory_db.expire_all()
    player = in_memory_db.query(PlayerProjection).filter_by(player_id="teamless").first()
    assert player.team == "NYY"


def test_batter_and_pitcher_position_sets():
    """Verify position classification sets match fantasy baseball conventions."""
    assert "OF" in BATTER_POS
    assert "P" in PITCHER_POS
    assert "SP" in PITCHER_POS
    assert "RP" in PITCHER_POS
    assert "DH" in BATTER_POS
    assert "UTIL" in BATTER_POS

    # No overlap between sets
    assert BATTER_POS.isdisjoint(PITCHER_POS)


def test_compute_cat_scores_single_player():
    """Edge case: single player returns z=0 (no pool to compare)."""
    players = [{
        "player_id": "solo",
        "proj": {
            "r": 80, "h": 140, "hr": 25, "rbi": 75, "k_bat": 90,
            "tb": 230, "avg": 0.270, "ops": 0.800, "nsb": 8,
        },
        "cat_scores": {},
    }]

    compute_cat_scores(players, BATTER_WEIGHTS)

    # All z-scores should be 0 (single player pool)
    for z in players[0]["cat_scores"].values():
        assert z == 0.0
    assert players[0]["z_score"] == 0.0


def test_compute_cat_scores_empty_list():
    """Edge case: empty player list does not crash."""
    compute_cat_scores([], BATTER_WEIGHTS)  # Should not raise


def test_run_backfill_empty_database(in_memory_db):
    """Edge case: no rows in player_projections."""
    result = run_backfill(in_memory_db)
    assert result["status"] == "success"
    assert result["cat_scores_updated"] == 0
    assert result["verify_remaining_empty"] == 0
